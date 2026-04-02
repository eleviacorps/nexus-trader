from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_model_artifacts(
    *,
    training_summary_path: Path,
    walkforward_report_path: Path,
    manifest_path: Path | None = None,
    precision_gate_path: Path | None = None,
    meta_gate_path: Path | None = None,
    gate_context_path: Path | None = None,
    timestamps_path: Path | None = None,
    targets_multihorizon_path: Path | None = None,
) -> dict[str, Any]:
    training = _load_json(training_summary_path)
    walkforward = _load_json(walkforward_report_path)
    manifest = _load_json(manifest_path) if manifest_path is not None and manifest_path.exists() else {}
    precision_gate = _load_json(precision_gate_path) if precision_gate_path is not None and precision_gate_path.exists() else {}
    gate_context_shape = None
    timestamps_len = None
    target_lengths: dict[str, int] = {}
    if gate_context_path is not None and gate_context_path.exists():
        gate_context_shape = list(np.load(gate_context_path, mmap_mode="r").shape)
    if timestamps_path is not None and timestamps_path.exists():
        timestamps_len = int(len(np.load(timestamps_path, mmap_mode="r")))
    if targets_multihorizon_path is not None and targets_multihorizon_path.exists():
        bundle = np.load(targets_multihorizon_path)
        target_lengths = {key: int(len(bundle[key])) for key in bundle.files}
    meta_gate_available = False
    meta_gate_provider = ""
    if meta_gate_path is not None and meta_gate_path.exists():
        try:
            import pickle

            with meta_gate_path.open("rb") as handle:
                meta_gate = pickle.load(handle)
            meta_gate_available = bool(meta_gate.get("available", False))
            meta_gate_provider = str(meta_gate.get("provider", ""))
        except Exception:
            meta_gate_available = True
            meta_gate_provider = "unknown"

    training_test = training.get("test_metrics", {})
    training_val = training.get("val_metrics", {})
    walk_overall = walkforward.get("overall", {})
    train_primary_roc = float(((training_test.get("primary", {}) or {}).get("roc_auc", 0.0)) or 0.0)
    val_primary_roc = float(((training_val.get("primary", {}) or {}).get("roc_auc", 0.0)) or 0.0)
    strategic_roc = float(((walk_overall.get("calibrated_metrics", {}) or {}).get("roc_auc", 0.0)) or 0.0)
    gate_train_accuracy = float((precision_gate.get("train_accuracy", 0.0)) or 0.0)
    gate_train_precision = float((precision_gate.get("train_precision", 0.0)) or 0.0)
    gate_positive_rate = float((walk_overall.get("gate_positive_rate", 0.0)) or 0.0)

    findings = []
    if gate_train_accuracy > 0.85 and gate_train_precision < 0.2:
        findings.append("precision_gate_has_high_train_accuracy_but_low_train_precision")
    if abs(val_primary_roc - train_primary_roc) > 0.05:
        findings.append("validation_test_roc_gap_is_large")
    if gate_positive_rate < 0.02:
        findings.append("gate_participation_near_zero")
    if gate_context_shape is not None and timestamps_len is not None and gate_context_shape[0] != timestamps_len:
        findings.append("gate_context_length_differs_from_timestamp_length")
    if target_lengths:
        unique_lengths = set(target_lengths.values())
        if len(unique_lengths) > 1:
            findings.append("multihorizon_target_lengths_mismatch")

    return {
        "training_summary_path": str(training_summary_path),
        "walkforward_report_path": str(walkforward_report_path),
        "manifest_path": str(manifest_path) if manifest_path is not None else "",
        "precision_gate_path": str(precision_gate_path) if precision_gate_path is not None else "",
        "meta_gate_path": str(meta_gate_path) if meta_gate_path is not None else "",
        "gate_context_path": str(gate_context_path) if gate_context_path is not None else "",
        "timestamps_path": str(timestamps_path) if timestamps_path is not None else "",
        "targets_multihorizon_path": str(targets_multihorizon_path) if targets_multihorizon_path is not None else "",
        "manifest_run_tag": str(manifest.get("run_tag", "")),
        "manifest_checkpoint": str(manifest.get("checkpoint_path", "")),
        "train_primary_roc_auc": train_primary_roc,
        "val_primary_roc_auc": val_primary_roc,
        "walkforward_calibrated_roc_auc": strategic_roc,
        "precision_gate_train_accuracy": gate_train_accuracy,
        "precision_gate_train_precision": gate_train_precision,
        "gate_positive_rate": gate_positive_rate,
        "meta_gate_available": meta_gate_available,
        "meta_gate_provider": meta_gate_provider,
        "gate_context_shape": gate_context_shape,
        "timestamps_len": timestamps_len,
        "target_lengths": target_lengths,
        "findings": findings,
    }
