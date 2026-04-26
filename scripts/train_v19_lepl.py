from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from config.project_config import V13_PAPER_TRADE_LOG_PATH, V14_PAPER_TRADE_LOG_PATH, V19_LEPL_MODEL_PATH, V19_LEPL_REPORT_PATH
from src.v19.lepl import LiveExecutionPolicy


def _iter_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _sqt_from_scores(uts_score: float, cabr_score: float) -> str:
    blend = 0.5 * float(uts_score) + 0.5 * float(cabr_score)
    if blend < 0.35:
        return "COLD"
    if blend < 0.55:
        return "NEUTRAL"
    if blend < 0.70:
        return "GOOD"
    return "HOT"


def _confidence_from_scores(uts_score: float, cabr_score: float) -> str:
    blend = 0.45 * float(uts_score) + 0.55 * float(cabr_score)
    if blend < 0.25:
        return "VERY_LOW"
    if blend < 0.45:
        return "LOW"
    if blend < 0.70:
        return "MODERATE"
    return "HIGH"


def _build_rows() -> tuple[np.ndarray, np.ndarray]:
    samples: list[dict] = []
    labels: list[str] = []
    for path in (V13_PAPER_TRADE_LOG_PATH, V14_PAPER_TRADE_LOG_PATH):
        for row in _iter_jsonl(path):
            uts = float(row.get("uts_score", 0.5) or 0.5)
            cabr = float(row.get("cabr_score", 0.5) or 0.5)
            pnl = float(row.get("pnl_pips", 0.0) or 0.0)
            stance = str(row.get("direction", "HOLD")).upper()
            hurst = 0.15 if stance == "BUY" else -0.15 if stance == "SELL" else 0.0
            disagreement = max(0.0, 1.0 - cabr)
            cpm_score = max(0.0, min(1.0, uts))
            base = {
                "sjd_stance": stance,
                "sjd_confidence": _confidence_from_scores(uts, cabr),
                "sqt_label": _sqt_from_scores(uts, cabr),
                "cabr_score": cabr,
                "hurst_asymmetry": hurst,
                "mfg_disagreement": disagreement,
                "cpm_score": cpm_score,
                "has_open_position": False,
                "open_position_pnl": 0.0,
            }
            samples.append(base)
            labels.append("ENTER")

            hold_row = dict(base)
            hold_row["has_open_position"] = True
            hold_row["open_position_pnl"] = max(pnl * 0.35, 1.0)
            samples.append(hold_row)
            labels.append("HOLD")

            close_row = dict(base)
            close_row["has_open_position"] = True
            close_row["open_position_pnl"] = pnl if pnl != 0.0 else -5.0
            close_row["sqt_label"] = "COLD" if pnl < 0.0 else close_row["sqt_label"]
            samples.append(close_row)
            labels.append("CLOSE")

            nothing_row = dict(base)
            nothing_row["sjd_stance"] = "HOLD"
            nothing_row["sjd_confidence"] = "VERY_LOW"
            nothing_row["sqt_label"] = "COLD"
            nothing_row["cabr_score"] = min(cabr, 0.25)
            nothing_row["cpm_score"] = min(cpm_score, 0.25)
            nothing_row["mfg_disagreement"] = max(disagreement, 0.65)
            samples.append(nothing_row)
            labels.append("NOTHING")
    if not samples:
        raise RuntimeError("No V13/V14 paper-trade rows were available to build LEPL training data.")
    policy = LiveExecutionPolicy()
    X = np.asarray([policy._features_to_vector(item) for item in samples], dtype=np.float32)
    y = np.asarray(labels)
    return X, y


def train_lepl(*, report_path: Path = V19_LEPL_REPORT_PATH, model_path: Path = V19_LEPL_MODEL_PATH) -> dict[str, object]:
    X, y = _build_rows()
    rng = np.random.default_rng(42)
    indices = np.arange(len(X))
    rng.shuffle(indices)
    cutoff = max(1, int(len(indices) * 0.8))
    train_idx = indices[:cutoff]
    valid_idx = indices[cutoff:]
    policy = LiveExecutionPolicy()
    policy.fit(X[train_idx], y[train_idx])
    predictions = np.asarray([policy.model.predict([row])[0] for row in X[valid_idx]])
    accuracy = float(np.mean(predictions == y[valid_idx])) if len(valid_idx) else 0.0
    policy.save(model_path)
    report = {
        "model_path": str(model_path),
        "train_rows": int(len(train_idx)),
        "valid_rows": int(len(valid_idx)),
        "accuracy": accuracy,
        "class_counts": {str(label): int(np.sum(y == label)) for label in sorted(set(y.tolist()))},
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the V19 LEPL execution policy from V13/V14 paper outcomes.")
    parser.parse_args()
    report = train_lepl()
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
