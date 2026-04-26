from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.project_config import (
    AMP_DTYPE,
    AMP_ENABLED,
    FUSED_FEATURE_MATRIX_PATH,
    FUSED_TIMESTAMPS_PATH,
    GATE_CONTEXT_PATH,
    META_GATE_PATH,
    MODEL_MANIFEST_PATH,
    NUM_WORKERS,
    PIN_MEMORY,
    PREFETCH_FACTOR,
    PRECISION_GATE_PATH,
    PERSISTENT_WORKERS,
    TARGETS_MULTIHORIZON_PATH,
    TFT_MODEL_DIR,
)
from scripts.train_fused_tft import tagged_path
from src.data.fused_dataset import DatasetSlice
from src.evaluation.walkforward import load_model, predict_multihorizon_for_slice
from src.training.meta_gate import (
    apply_meta_gate,
    combine_gate_scores,
    save_meta_gate,
    train_meta_gate,
)
from src.training.train_tft import (
    apply_precision_gate,
    build_calibration_report,
    find_optimal_threshold,
    train_precision_gate,
)
from src.utils.training_splits import split_by_years


def parse_year_list(values: Any) -> list[int]:
    if isinstance(values, (list, tuple)):
        return [int(value) for value in values]
    if isinstance(values, str):
        return [int(part.strip()) for part in values.split(",") if part.strip()]
    return []


def resolve_run_manifest(run_tag: str) -> Path:
    if run_tag:
        candidate = TFT_MODEL_DIR / f"model_manifest_{run_tag}.json"
        if candidate.exists():
            return candidate
    return MODEL_MANIFEST_PATH


def slice_gate_context(path: Path, row_slice: DatasetSlice, sequence_len: int) -> np.ndarray | None:
    if not path.exists():
        return None
    context = np.load(path, mmap_mode="r")
    start = row_slice.start + sequence_len - 1
    stop = row_slice.stop + sequence_len - 1
    return np.asarray(context[start:stop], dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate precision/meta gate artifacts from an existing checkpoint.")
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--amp", dest="amp_enabled", action="store_true", default=AMP_ENABLED)
    parser.add_argument("--no-amp", dest="amp_enabled", action="store_false")
    parser.add_argument("--amp-dtype", default=AMP_DTYPE, choices=["bfloat16", "float16", "none"])
    args = parser.parse_args()

    manifest_path = resolve_run_manifest(args.run_tag)
    model, manifest, device = load_model(manifest_path=manifest_path)
    if not bool(manifest.get("multi_horizon", False)):
        raise ValueError("Gate regeneration currently supports multi-horizon manifests only.")

    horizon_labels = list(manifest.get("horizon_labels", ["5m", "10m", "15m", "30m"]))
    output_labels = list(manifest.get("output_labels", []))
    if not output_labels:
        raise ValueError("Manifest is missing output_labels.")
    sequence_len = int(manifest.get("sequence_len", 120))

    timestamps = np.load(FUSED_TIMESTAMPS_PATH, mmap_mode="r")
    feature_rows = int(np.load(FUSED_FEATURE_MATRIX_PATH, mmap_mode="r").shape[0])
    split_report = manifest.get("split_report", {})
    train_years = parse_year_list(split_report.get("train_years", []))
    val_years = parse_year_list(split_report.get("val_years", []))
    test_years = parse_year_list(split_report.get("test_years", []))
    split_config = split_by_years(timestamps[:feature_rows], sequence_len, train_years, val_years, test_years)
    val_slice = split_config.val
    test_slice = split_config.test

    amp_enabled = bool(args.amp_enabled and str(args.amp_dtype).lower() != "none")
    effective_amp_dtype = "none" if not amp_enabled else str(args.amp_dtype).lower()

    val_targets, val_probabilities = predict_multihorizon_for_slice(
        model,
        device,
        val_slice,
        feature_path=FUSED_FEATURE_MATRIX_PATH,
        target_bundle_path=TARGETS_MULTIHORIZON_PATH,
        target_keys=output_labels,
        sequence_len=sequence_len,
        batch_size=args.batch_size,
        amp_enabled=amp_enabled,
        amp_dtype=effective_amp_dtype,
    )
    test_targets, test_probabilities = predict_multihorizon_for_slice(
        model,
        device,
        test_slice,
        feature_path=FUSED_FEATURE_MATRIX_PATH,
        target_bundle_path=TARGETS_MULTIHORIZON_PATH,
        target_keys=output_labels,
        sequence_len=sequence_len,
        batch_size=args.batch_size,
        amp_enabled=amp_enabled,
        amp_dtype=effective_amp_dtype,
    )

    threshold_selection = find_optimal_threshold(val_targets[:, 2], val_probabilities[:, 2], metric="f1")
    threshold = float(threshold_selection["threshold"])
    gate_context_val = slice_gate_context(GATE_CONTEXT_PATH, val_slice, sequence_len)
    gate_context_test = slice_gate_context(GATE_CONTEXT_PATH, test_slice, sequence_len)

    precision_gate = train_precision_gate(
        val_probabilities,
        val_targets,
        context_features=gate_context_val,
        threshold=threshold,
    )
    precision_gate_probabilities_val = apply_precision_gate(val_probabilities, precision_gate, context_features=gate_context_val)
    precision_gate_probabilities_test = apply_precision_gate(test_probabilities, precision_gate, context_features=gate_context_test)

    meta_gate = train_meta_gate(
        val_probabilities,
        val_targets,
        context_features=gate_context_val,
        threshold=threshold,
    )
    meta_gate_probabilities_val = apply_meta_gate(val_probabilities, meta_gate, context_features=gate_context_val)
    meta_gate_probabilities_test = apply_meta_gate(test_probabilities, meta_gate, context_features=gate_context_test)
    combined_gate_val = combine_gate_scores(precision_gate_probabilities_val, meta_gate_probabilities_val)
    combined_gate_test = combine_gate_scores(precision_gate_probabilities_test, meta_gate_probabilities_test)

    precision_gate_path = tagged_path(PRECISION_GATE_PATH, args.run_tag)
    meta_gate_path = tagged_path(META_GATE_PATH, args.run_tag)
    precision_gate_path.parent.mkdir(parents=True, exist_ok=True)
    precision_gate_path.write_text(json.dumps(precision_gate, indent=2), encoding="utf-8")
    if meta_gate is not None and meta_gate.get("available", False):
        save_meta_gate(meta_gate_path, meta_gate)

    gate_report = {
        "run_tag": args.run_tag,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "precision_gate_path": str(precision_gate_path),
        "meta_gate_path": str(meta_gate_path) if meta_gate is not None and meta_gate.get("available", False) else "",
        "threshold_selection": threshold_selection,
        "precision_gate": {
            "threshold": float(precision_gate.get("threshold", 0.0)),
            "train_participation": float(precision_gate.get("train_participation", 0.0)),
            "train_precision": float(precision_gate.get("train_precision", 0.0)),
            "positive_rate": float(precision_gate.get("positive_rate", 0.0)),
            "tradeable_rate": float(precision_gate.get("tradeable_rate", 0.0)),
            "quant_tradeability_mean": precision_gate.get("quant_tradeability_mean"),
        },
        "meta_gate": {
            "available": bool(meta_gate.get("available", False)) if meta_gate is not None else False,
            "provider": str(meta_gate.get("provider", "")) if meta_gate is not None else "",
            "threshold": float(meta_gate.get("threshold", 0.0)) if meta_gate is not None and meta_gate.get("available", False) else 0.0,
            "train_participation": float(meta_gate.get("train_participation", 0.0)) if meta_gate is not None and meta_gate.get("available", False) else 0.0,
            "train_precision": float(meta_gate.get("train_precision", 0.0)) if meta_gate is not None and meta_gate.get("available", False) else 0.0,
        },
        "validation_curves": {
            "precision_gate": build_calibration_report(
                ((val_probabilities[:, 2] >= threshold).astype(np.float32) == val_targets[:, 2].astype(np.float32)).astype(np.float32),
                precision_gate_probabilities_val,
            ),
            "meta_gate": build_calibration_report(
                ((val_probabilities[:, 2] >= threshold).astype(np.float32) == val_targets[:, 2].astype(np.float32)).astype(np.float32),
                meta_gate_probabilities_val if meta_gate_probabilities_val is not None else precision_gate_probabilities_val,
            ),
            "combined_gate": build_calibration_report(
                ((val_probabilities[:, 2] >= threshold).astype(np.float32) == val_targets[:, 2].astype(np.float32)).astype(np.float32),
                combined_gate_val if combined_gate_val is not None else precision_gate_probabilities_val,
            ),
        },
        "test_curves": {
            "precision_gate": build_calibration_report(
                ((test_probabilities[:, 2] >= threshold).astype(np.float32) == test_targets[:, 2].astype(np.float32)).astype(np.float32),
                precision_gate_probabilities_test,
            ),
            "meta_gate": build_calibration_report(
                ((test_probabilities[:, 2] >= threshold).astype(np.float32) == test_targets[:, 2].astype(np.float32)).astype(np.float32),
                meta_gate_probabilities_test if meta_gate_probabilities_test is not None else precision_gate_probabilities_test,
            ),
            "combined_gate": build_calibration_report(
                ((test_probabilities[:, 2] >= threshold).astype(np.float32) == test_targets[:, 2].astype(np.float32)).astype(np.float32),
                combined_gate_test if combined_gate_test is not None else precision_gate_probabilities_test,
            ),
        },
    }
    report_path = Path("outputs") / "evaluation" / f"gate_regeneration_report_{args.run_tag}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(gate_report, indent=2), encoding="utf-8")
    print(json.dumps(gate_report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
