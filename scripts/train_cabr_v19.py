from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from config.project_config import V19_BRANCH_ARCHIVE_PATH, V19_CABR_BASE_MODEL_PATH, V19_CABR_MODEL_PATH, V19_CABR_TRAINING_REPORT_PATH
from src.v13.cabr import (
    CABR,
    _batch_to_device,
    _pair_dataloader,
    augment_cabr_context,
    derive_cabr_feature_columns,
    evaluate_cabr_pairwise_accuracy,
)
from src.v19.curriculum_pairs import build_curriculum_pair_payload


def _resolve_feature_columns(frame: pd.DataFrame) -> tuple[tuple[str, ...], tuple[str, ...]]:
    branch_cols, context_cols = derive_cabr_feature_columns(frame)
    branch_extras = (
        "analog_confidence",
        "cone_realism",
        "mfg_disagreement",
        "volatility_realism",
        "fair_value_dislocation",
        "contradiction_full_agreement",
        "contradiction_partial_disagreement",
        "contradiction_full_disagreement",
        "contradiction_mixed",
        "wltc_state_retail_dominant",
        "wltc_state_institutional_dominant",
        "wltc_state_balanced",
    )
    context_extras = (
        "quant_regime_strength",
        "quant_transition_risk",
        "quant_vol_realism",
        "quant_fair_value_z",
        "quant_route_confidence",
        "quant_trend_score",
        "mfg_disagreement",
        "hurst_overall",
        "hurst_positive",
        "hurst_negative",
        "hurst_asymmetry",
    )
    branch_cols = tuple(dict.fromkeys(tuple(branch_cols) + tuple(column for column in branch_extras if column in frame.columns)))
    context_cols = tuple(dict.fromkeys(tuple(context_cols) + tuple(column for column in context_extras if column in frame.columns)))
    return branch_cols, context_cols


def _time_split(frame: pd.DataFrame, validation_fraction: float = 0.20) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = frame.copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True, errors="coerce")
    working = working.sort_values("timestamp").reset_index(drop=True)
    cutoff = max(1, int(len(working) * (1.0 - float(validation_fraction))))
    return working.iloc[:cutoff].copy(), working.iloc[cutoff:].copy()


def _train_stage(
    model: CABR,
    train_payload: dict[str, Any],
    valid_payload: dict[str, Any],
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
) -> tuple[CABR, dict[str, Any]]:
    target_device = torch.device(device)
    model = model.to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loader = _pair_dataloader(train_payload, batch_size=batch_size, shuffle=True)
    best_accuracy = -1.0
    best_state = None
    history: list[dict[str, float]] = []
    for _ in range(max(int(epochs), 1)):
        model.train()
        epoch_loss = 0.0
        steps = 0
        for batch in loader:
            prepared = _batch_to_device(batch, target_device)
            score_a = model.score(prepared.branch_a, prepared.context_a)
            score_b = model.score(prepared.branch_b, prepared.context_b)
            loss = model.pairwise_loss(score_a, score_b, prepared.label)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            steps += 1
        metrics = evaluate_cabr_pairwise_accuracy(model, valid_payload, device=device)
        valid_acc = float(metrics["overall_accuracy"])
        history.append({"loss": epoch_loss / max(steps, 1), "valid_accuracy": valid_acc})
        if valid_acc > best_accuracy:
            best_accuracy = valid_acc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_accuracy": best_accuracy, "history": history}


def train_cabr_v19(
    *,
    archive_path: Path = V19_BRANCH_ARCHIVE_PATH,
    base_checkpoint_path: Path = V19_CABR_BASE_MODEL_PATH,
    final_checkpoint_path: Path = V19_CABR_MODEL_PATH,
    report_path: Path = V19_CABR_TRAINING_REPORT_PATH,
    warmup_epochs: int = 50,
    full_epochs: int = 200,
    regime_epochs: int = 20,
    batch_size: int = 1024,
    lr: float = 3e-4,
    device: str | None = None,
    max_rows: int | None = None,
    max_pairs: int = 100_000,
    max_valid_pairs: int = 40_000,
) -> dict[str, Any]:
    frame = pd.read_parquet(archive_path)
    if max_rows is not None and int(max_rows) > 0 and len(frame) > int(max_rows):
        frame = frame.sort_values("timestamp").tail(int(max_rows)).reset_index(drop=True)
    alias_map = {
        "branch_disagreement": "analog_disagreement",
        "consensus_strength": "consensus_score",
        "analog_disagreement_v9": "analog_disagreement",
        "crowd_consistency_v9": "crowd_consistency",
        "news_consistency_v9": "news_consistency",
        "macro_consistency_v9": "macro_alignment",
    }
    for target, source in alias_map.items():
        if target not in frame.columns:
            frame[target] = frame[source] if source in frame.columns else 0.0
    if "setl_target_net_unit_pnl" not in frame.columns:
        frame["setl_target_net_unit_pnl"] = frame.get("actual_final_return", 0.0)
    frame = augment_cabr_context(frame, mmm_features=None)
    train_frame, valid_frame = _time_split(frame)
    branch_cols, context_cols = _resolve_feature_columns(train_frame)
    for column in branch_cols:
        if column not in train_frame.columns:
            train_frame[column] = 0.0
        if column not in valid_frame.columns:
            valid_frame[column] = 0.0
    for column in context_cols:
        if column not in train_frame.columns:
            train_frame[column] = 0.0
        if column not in valid_frame.columns:
            valid_frame[column] = 0.0

    curriculum = build_curriculum_pair_payload(
        train_frame,
        branch_feature_names=branch_cols,
        context_feature_names=context_cols,
        use_temporal_context=True,
        n_context_bars=12,
        max_pairs=int(max_pairs),
        easy_quantile=0.65,
    )
    valid_payload = build_curriculum_pair_payload(
        valid_frame,
        branch_feature_names=branch_cols,
        context_feature_names=context_cols,
        use_temporal_context=True,
        n_context_bars=12,
        max_pairs=int(max_valid_pairs),
        easy_quantile=0.65,
    )["full"]
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = CABR(
        branch_feature_dim=len(branch_cols),
        context_feature_dim=len(context_cols),
        use_temporal_context=True,
        n_context_bars=12,
        use_chaotic_activation=True,
    )
    model, warmup_report = _train_stage(
        model,
        curriculum["easy"],
        valid_payload,
        epochs=int(warmup_epochs),
        batch_size=min(int(batch_size), 512),
        lr=float(lr),
        device=target_device,
    )
    base_payload = {
        "state_dict": model.state_dict(),
        "branch_feature_names": tuple(branch_cols),
        "context_feature_names": tuple(context_cols),
        "embed_dim": 64,
        "n_heads": 4,
        "use_temporal_context": True,
        "n_context_bars": 12,
        "use_chaotic_activation": True,
        "best_accuracy": warmup_report["best_accuracy"],
    }
    base_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base_payload, base_checkpoint_path)

    model, full_report = _train_stage(
        model,
        curriculum["full"],
        valid_payload,
        epochs=int(full_epochs),
        batch_size=int(batch_size),
        lr=float(lr),
        device=target_device,
    )

    regime_reports: dict[str, Any] = {}
    for regime in ("trend_up", "trend_down", "range", "breakout"):
        subset = train_frame.loc[train_frame["regime_class"].astype(str) == regime].copy()
        valid_subset = valid_frame.loc[valid_frame["regime_class"].astype(str) == regime].copy()
        if len(subset) < 32 or len(valid_subset) < 16:
            continue
        subset_payload = build_curriculum_pair_payload(
            subset,
            branch_feature_names=branch_cols,
            context_feature_names=context_cols,
            use_temporal_context=True,
            n_context_bars=12,
            max_pairs=max(5_000, int(max_pairs) // 5),
            easy_quantile=0.55,
        )["full"]
        valid_regime_payload = build_curriculum_pair_payload(
            valid_subset,
            branch_feature_names=branch_cols,
            context_feature_names=context_cols,
            use_temporal_context=True,
            n_context_bars=12,
            max_pairs=max(2_000, int(max_valid_pairs) // 5),
            easy_quantile=0.55,
        )["full"]
        model, stage_report = _train_stage(
            model,
            subset_payload,
            valid_regime_payload,
            epochs=int(regime_epochs),
            batch_size=min(int(batch_size), 512),
            lr=float(lr) * 0.5,
            device=target_device,
        )
        regime_reports[regime] = stage_report

    final_metrics = evaluate_cabr_pairwise_accuracy(model, valid_payload, device=target_device)
    final_payload = {
        "state_dict": model.state_dict(),
        "branch_feature_names": tuple(branch_cols),
        "context_feature_names": tuple(context_cols),
        "embed_dim": 64,
        "n_heads": 4,
        "use_temporal_context": True,
        "n_context_bars": 12,
        "use_chaotic_activation": True,
        "best_accuracy": float(final_metrics["overall_accuracy"]),
        "warmup_accuracy": warmup_report["best_accuracy"],
        "full_stage_accuracy": full_report["best_accuracy"],
        "regime_reports": regime_reports,
    }
    final_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(final_payload, final_checkpoint_path)
    report = {
        "archive_path": str(archive_path),
        "base_checkpoint_path": str(base_checkpoint_path),
        "final_checkpoint_path": str(final_checkpoint_path),
        "warmup": warmup_report,
        "full_stage": full_report,
        "regime_stage": regime_reports,
        "final_metrics": final_metrics,
        "train_rows": int(len(train_frame)),
        "valid_rows": int(len(valid_frame)),
        "easy_pairs": int(len(curriculum["easy"].get("pairs", []))),
        "full_pairs": int(len(curriculum["full"].get("pairs", []))),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the V19 CABR recovery stack with curriculum and regime fine-tuning.")
    parser.add_argument("--warmup-epochs", type=int, default=50)
    parser.add_argument("--full-epochs", type=int, default=200)
    parser.add_argument("--regime-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=100000)
    parser.add_argument("--max-valid-pairs", type=int, default=40000)
    args = parser.parse_args()
    report = train_cabr_v19(
        warmup_epochs=int(args.warmup_epochs),
        full_epochs=int(args.full_epochs),
        regime_epochs=int(args.regime_epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        device=args.device,
        max_rows=int(args.max_rows) if int(args.max_rows) > 0 else None,
        max_pairs=int(args.max_pairs),
        max_valid_pairs=int(args.max_valid_pairs),
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
