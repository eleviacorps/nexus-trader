from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.project_config import V12_SARV_REPORT_PATH
from src.v12.tctl import TCTLRanker, evaluate_stage_policy, optimize_tctl_threshold, score_tctl_model


@dataclass(frozen=True)
class SarvStageResult:
    win_rate: float | None
    avg_unit_pnl: float | None
    participation: float | None
    trade_count: int
    passed: bool
    status: str = "complete"
    gap_vs_stage1_winrate: float | None = None


def _result_from_policy(policy: dict[str, Any], *, passed: bool, gap_vs_stage1_winrate: float | None = None) -> SarvStageResult:
    return SarvStageResult(
        win_rate=float(policy.get("win_rate", 0.0)),
        avg_unit_pnl=float(policy.get("avg_unit_pnl", 0.0)),
        participation=float(policy.get("participation", 0.0)),
        trade_count=int(policy.get("trade_count", 0)),
        passed=bool(passed),
        gap_vs_stage1_winrate=None if gap_vs_stage1_winrate is None else float(gap_vs_stage1_winrate),
    )


def run_sarv_validation(
    *,
    model: TCTLRanker,
    train_candidates: pd.DataFrame,
    archive_candidates: pd.DataFrame,
    feature_names: tuple[str, ...],
    bar_replay_candidates: pd.DataFrame | None = None,
    paper_trade_log_path: Path | None = None,
) -> dict[str, Any]:
    train_scores = score_tctl_model(model, train_candidates, feature_names=feature_names)
    threshold = optimize_tctl_threshold(train_scores, train_candidates["setl_target_net_unit_pnl"].to_numpy(dtype=np.float32))

    archive_frame = archive_candidates.copy()
    archive_frame["tctl_score"] = score_tctl_model(model, archive_frame, feature_names=feature_names)
    stage_1_policy = evaluate_stage_policy(archive_frame, score_column="tctl_score", threshold=threshold.threshold)
    stage_1_passed = (
        stage_1_policy["win_rate"] > 0.55
        and stage_1_policy["avg_unit_pnl"] > 0.10
        and 0.15 <= stage_1_policy["participation"] <= 0.45
    )
    stage_1 = _result_from_policy(stage_1_policy, passed=stage_1_passed)

    replay_frame = (bar_replay_candidates if bar_replay_candidates is not None else archive_candidates).copy()
    replay_frame["tctl_score"] = score_tctl_model(model, replay_frame, feature_names=feature_names)
    stage_2_policy = evaluate_stage_policy(replay_frame, score_column="tctl_score", threshold=threshold.threshold)
    gap = abs(float(stage_2_policy["win_rate"]) - float(stage_1_policy["win_rate"]))
    stage_2_passed = gap <= 0.05 and float(stage_2_policy["avg_unit_pnl"]) > 0.0
    stage_2 = _result_from_policy(stage_2_policy, passed=stage_2_passed, gap_vs_stage1_winrate=gap)

    stage_3: dict[str, Any]
    if paper_trade_log_path is None or not paper_trade_log_path.exists():
        stage_3 = {
            "win_rate": None,
            "avg_unit_pnl": None,
            "participation": None,
            "trade_count": 0,
            "passed": False,
            "status": "pending",
        }
    else:
        trades = pd.read_json(paper_trade_log_path, lines=True)
        trades = trades.loc[trades.get("status", "complete") == "complete"].copy()
        won_mask = trades.get("outcome", pd.Series(dtype=object)).astype(str).str.lower().eq("win")
        win_rate = float(won_mask.mean()) if len(trades) else 0.0
        trade_count = int(len(trades))
        participation = float(trades.get("participation", pd.Series([0.25] * max(trade_count, 1))).mean()) if len(trades) else 0.0
        avg_unit_pnl = float(trades.get("pnl_pips", pd.Series([0.0] * max(trade_count, 1))).mean()) if len(trades) else 0.0
        stage_3 = {
            "win_rate": round(win_rate, 6),
            "avg_unit_pnl": round(avg_unit_pnl, 6),
            "participation": round(participation, 6),
            "trade_count": trade_count,
            "passed": bool(win_rate > 0.52 and trade_count >= 40 and 0.15 <= participation <= 0.40),
            "status": "complete",
        }

    summary = {
        "model": "tctl_v12",
        "threshold": {
            "threshold": round(float(threshold.threshold), 6),
            "train_participation": round(float(threshold.participation_rate), 6),
            "train_avg_unit_pnl": round(float(threshold.avg_unit_pnl), 6),
        },
        "stage_1": asdict(stage_1),
        "stage_2": asdict(stage_2),
        "stage_3": stage_3,
        "overall_passed": bool(stage_1_passed and stage_2_passed and stage_3.get("passed", False)),
    }
    return summary


def run_scored_sarv_validation(
    *,
    model_name: str,
    train_scores: np.ndarray,
    train_outcomes: np.ndarray,
    archive_candidates: pd.DataFrame,
    archive_score_column: str,
    bar_replay_candidates: pd.DataFrame | None = None,
    replay_score_column: str | None = None,
    paper_trade_log_path: Path | None = None,
) -> dict[str, Any]:
    threshold = optimize_tctl_threshold(np.asarray(train_scores, dtype=np.float32), np.asarray(train_outcomes, dtype=np.float32))

    archive_frame = archive_candidates.copy()
    stage_1_policy = evaluate_stage_policy(archive_frame, score_column=archive_score_column, threshold=threshold.threshold)
    stage_1_passed = (
        stage_1_policy["win_rate"] > 0.55
        and stage_1_policy["avg_unit_pnl"] > 0.10
        and 0.15 <= stage_1_policy["participation"] <= 0.45
    )
    stage_1 = _result_from_policy(stage_1_policy, passed=stage_1_passed)

    replay_frame = (bar_replay_candidates if bar_replay_candidates is not None else archive_candidates).copy()
    replay_column = replay_score_column or archive_score_column
    stage_2_policy = evaluate_stage_policy(replay_frame, score_column=replay_column, threshold=threshold.threshold)
    gap = abs(float(stage_2_policy["win_rate"]) - float(stage_1_policy["win_rate"]))
    stage_2_passed = gap <= 0.05 and float(stage_2_policy["avg_unit_pnl"]) > 0.0
    stage_2 = _result_from_policy(stage_2_policy, passed=stage_2_passed, gap_vs_stage1_winrate=gap)

    stage_3: dict[str, Any]
    if paper_trade_log_path is None or not paper_trade_log_path.exists():
        stage_3 = {
            "win_rate": None,
            "avg_unit_pnl": None,
            "participation": None,
            "trade_count": 0,
            "passed": False,
            "status": "pending",
        }
    else:
        trades = pd.read_json(paper_trade_log_path, lines=True)
        trades = trades.loc[trades.get("status", "complete") == "complete"].copy()
        won_mask = trades.get("outcome", pd.Series(dtype=object)).astype(str).str.lower().eq("win")
        win_rate = float(won_mask.mean()) if len(trades) else 0.0
        trade_count = int(len(trades))
        participation = float(trades.get("participation", pd.Series([0.25] * max(trade_count, 1))).mean()) if len(trades) else 0.0
        avg_unit_pnl = float(trades.get("pnl_pips", pd.Series([0.0] * max(trade_count, 1))).mean()) if len(trades) else 0.0
        stage_3 = {
            "win_rate": round(win_rate, 6),
            "avg_unit_pnl": round(avg_unit_pnl, 6),
            "participation": round(participation, 6),
            "trade_count": trade_count,
            "passed": bool(win_rate > 0.52 and trade_count >= 40 and 0.15 <= participation <= 0.40),
            "status": "complete",
        }

    return {
        "model": model_name,
        "threshold": {
            "threshold": round(float(threshold.threshold), 6),
            "train_participation": round(float(threshold.participation_rate), 6),
            "train_avg_unit_pnl": round(float(threshold.avg_unit_pnl), 6),
        },
        "stage_1": asdict(stage_1),
        "stage_2": asdict(stage_2),
        "stage_3": stage_3,
        "overall_passed": bool(stage_1_passed and stage_2_passed and stage_3.get("passed", False)),
    }


def write_sarv_report(report: dict[str, Any], path: Path = V12_SARV_REPORT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path
