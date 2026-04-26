from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.project_config import MODELS_DIR, OUTPUTS_DIR
from src.v25_1.branch_ensemble_ranker import BranchEnsembleRanker
from src.v25_1.regime_specific_branch_ranker import RegimeSpecificBranchRanker


@dataclass(frozen=True)
class BranchRealismReport:
    generated_at: str
    archive_path: str
    train_rows: int
    eval_rows: int
    baseline_top1_accuracy: float
    v25_1_top1_accuracy: float
    branch_realism_improvement_ratio: float
    branch_realism_improvement_pct: float
    target_branch_realism_improvement_pct: float
    target_reached: bool
    regime_model_summary: dict[str, Any]
    blend_formula: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class BranchRealismRetrainer:
    def __init__(
        self,
        archive_path: Path | None = None,
        model_path: Path | None = None,
        report_path: Path | None = None,
    ):
        self.archive_path = archive_path or (OUTPUTS_DIR / "v19" / "branch_archive_100k.parquet")
        self.model_path = model_path or (MODELS_DIR / "v25_1" / "regime_specific_branch_ranker.json")
        self.report_path = report_path or (OUTPUTS_DIR / "v25" / "branch_realism_report.json")
        self.ranker = RegimeSpecificBranchRanker()
        self.ensemble = BranchEnsembleRanker()

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    def _prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        working = frame.copy()
        working["sample_key"] = working["sample_id"].astype(str)
        working["cabr_score"] = pd.to_numeric(working.get("branch_confidence"), errors="coerce").fillna(0.5).clip(0.0, 1.0)
        working["analog_similarity"] = pd.to_numeric(working.get("analog_similarity"), errors="coerce").fillna(0.5).clip(0.0, 1.0)
        working["volatility_realism"] = pd.to_numeric(working.get("volatility_realism"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
        working["hmm_regime_match"] = pd.to_numeric(working.get("hmm_regime_match"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
        working["branch_disagreement"] = pd.to_numeric(working.get("branch_disagreement"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
        working["news_consistency"] = pd.to_numeric(working.get("news_consistency"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
        working["crowd_consistency"] = pd.to_numeric(working.get("crowd_consistency"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
        working["branch_volatility"] = pd.to_numeric(working.get("branch_volatility"), errors="coerce").fillna(0.0).clip(0.0, 8.0)
        working["branch_move_zscore_abs"] = pd.to_numeric(working.get("branch_move_zscore"), errors="coerce").fillna(0.0).abs().clip(0.0, 8.0)
        working["regime_bucket"] = working.get("dominant_regime", "unknown")
        working["actual_final_return"] = pd.to_numeric(working.get("actual_final_return"), errors="coerce").fillna(0.0)
        working["branch_move_size"] = pd.to_numeric(working.get("branch_move_size"), errors="coerce").fillna(0.0)
        working["label_hit"] = (
            np.sign(working["actual_final_return"].to_numpy(dtype=np.float64))
            == np.sign(working["branch_move_size"].to_numpy(dtype=np.float64))
        ).astype(float)
        return working

    @staticmethod
    def _top1_accuracy(frame: pd.DataFrame, score_col: str) -> float:
        if frame.empty:
            return 0.0
        top = (
            frame.sort_values(["sample_key", score_col], ascending=[True, False])
            .groupby("sample_key", as_index=False)
            .head(1)
        )
        if top.empty:
            return 0.0
        return float(np.mean(pd.to_numeric(top["label_hit"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)))

    def train_and_evaluate(self) -> BranchRealismReport:
        if not self.archive_path.exists():
            raise FileNotFoundError(f"Missing branch archive: {self.archive_path}")
        frame = pd.read_parquet(self.archive_path)
        prepared = self._prepare(frame)
        train = prepared.loc[prepared.get("year", 0) <= 2023].copy()
        evaluate = prepared.loc[prepared.get("year", 0) >= 2024].copy()
        if evaluate.empty:
            evaluate = prepared.tail(25000).copy()
            train = prepared.iloc[:-25000].copy()

        regime_summary = self.ranker.fit(train)
        self.ranker.save(self.model_path)

        # Historical expectancy of similar trades from train regime/direction/vol-bucket groups.
        train["vol_bucket"] = pd.qcut(train["branch_volatility"].rank(method="first"), 5, labels=False, duplicates="drop")
        evaluate["vol_bucket"] = pd.qcut(evaluate["branch_volatility"].rank(method="first"), 5, labels=False, duplicates="drop")
        hist_expectancy_map = (
            train.groupby(["regime_bucket", "branch_direction", "vol_bucket"], dropna=False)["actual_final_return"]
            .mean()
            .to_dict()
        )
        global_expectancy = float(train["actual_final_return"].mean()) if not train.empty else 0.0
        lower = float(np.percentile(train["actual_final_return"], 5)) if not train.empty else -0.01
        upper = float(np.percentile(train["actual_final_return"], 95)) if not train.empty else 0.01

        seq_realism = []
        hist_expectancy_norm = []
        for row in evaluate.to_dict(orient="records"):
            prediction = self.ranker.predict(row)
            seq_realism.append(prediction.probability)
            key = (row.get("regime_bucket"), row.get("branch_direction"), row.get("vol_bucket"))
            hist_value = self._safe_float(hist_expectancy_map.get(key, global_expectancy), global_expectancy)
            hist_expectancy_norm.append(float(np.clip((hist_value - lower) / max(upper - lower, 1e-6), 0.0, 1.0)))
        evaluate["sequence_realism_score"] = np.asarray(seq_realism, dtype=np.float64)
        evaluate["historical_expectancy_norm"] = np.asarray(hist_expectancy_norm, dtype=np.float64)

        ranked = self.ensemble.apply_dataframe(
            evaluate,
            cabr_col="cabr_score",
            sequence_realism_col="sequence_realism_score",
            analog_col="analog_similarity",
            historical_expectancy_col="historical_expectancy_norm",
            output_col="v25_1_rank_score",
        )
        baseline_accuracy = self._top1_accuracy(ranked, "cabr_score")
        v25_1_accuracy = self._top1_accuracy(ranked, "v25_1_rank_score")
        improvement_ratio = ((v25_1_accuracy - baseline_accuracy) / baseline_accuracy) if baseline_accuracy > 0.0 else 0.0
        report = BranchRealismReport(
            generated_at=datetime.now(tz=UTC).isoformat(),
            archive_path=str(self.archive_path),
            train_rows=int(len(train)),
            eval_rows=int(len(evaluate)),
            baseline_top1_accuracy=float(round(baseline_accuracy, 6)),
            v25_1_top1_accuracy=float(round(v25_1_accuracy, 6)),
            branch_realism_improvement_ratio=float(round(improvement_ratio, 6)),
            branch_realism_improvement_pct=float(round(improvement_ratio * 100.0, 3)),
            target_branch_realism_improvement_pct=15.0,
            target_reached=bool((improvement_ratio * 100.0) > 15.0),
            regime_model_summary=regime_summary,
            blend_formula="0.40*CABR + 0.25*sequence_realism + 0.20*analog_similarity + 0.15*historical_expectancy",
        )
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
        return report
