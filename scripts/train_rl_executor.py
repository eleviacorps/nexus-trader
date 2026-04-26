from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.project_config import OUTPUTS_V20_DIR, V20_FEATURES_PATH, V20_RL_HYPER_AGENT_PATH, V20_RL_SUBAGENTS_DIR
from src.v20.rl_executor import HierarchicalExecutor


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the V20 hierarchical RL fallback profiles.")
    parser.add_argument("--memory-len", type=int, default=20)
    args = parser.parse_args()

    if not V20_FEATURES_PATH.exists():
        raise SystemExit(f"Missing V20 feature frame at {V20_FEATURES_PATH}. Run build_v20_features.py first.")

    frame = pd.read_parquet(V20_FEATURES_PATH).replace([np.inf, -np.inf], np.nan).ffill().bfill()
    numeric = frame.select_dtypes(include=["number"]).copy()
    if "future_return_15m" not in numeric.columns:
        raise SystemExit("V20 feature frame is missing future_return_15m, so the RL fallback trainer cannot score outcomes.")

    executor = HierarchicalExecutor(memory_len=int(args.memory_len))
    V20_RL_SUBAGENTS_DIR.mkdir(parents=True, exist_ok=True)
    regime_reports: list[dict[str, object]] = []
    for regime in range(6):
        subset = numeric.loc[pd.to_numeric(numeric.get("hmm_state"), errors="coerce").fillna(-1).astype(int) == regime].copy()
        if subset.empty:
            continue
        mean_return = float(subset["future_return_15m"].mean())
        mean_abs_return = float(subset["future_return_15m"].abs().mean())
        action_bias = "BUY" if mean_return > 0 else "SELL" if mean_return < 0 else "HOLD"
        payload = {
            "regime": regime,
            "rows": int(len(subset)),
            "mean_return_15m": mean_return,
            "mean_abs_return_15m": mean_abs_return,
            "action_bias": action_bias,
        }
        torch.save(payload, V20_RL_SUBAGENTS_DIR / f"regime_{regime}.pt")
        regime_reports.append(payload)

    dominant_weights = {item["regime"]: round(abs(float(item["mean_return_15m"])) + float(item["mean_abs_return_15m"]), 6) for item in regime_reports}
    total_weight = sum(dominant_weights.values()) or 1.0
    hyper_payload = {
        "memory_len": int(args.memory_len),
        "regime_weights": {str(key): round(float(value / total_weight), 6) for key, value in dominant_weights.items()},
        "hyper_agent_state": executor.hyper_agent.state_dict(),
    }
    V20_RL_HYPER_AGENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(hyper_payload, V20_RL_HYPER_AGENT_PATH)
    report = {
        "regimes_trained": int(len(regime_reports)),
        "sub_agents_dir": str(V20_RL_SUBAGENTS_DIR),
        "hyper_agent_path": str(V20_RL_HYPER_AGENT_PATH),
        "regime_reports": regime_reports,
        "note": "This local RL phase is an offline regime-profile fallback, not a full PPO MacroHFT training run.",
    }
    report_path = OUTPUTS_V20_DIR / "rl_executor_training_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
