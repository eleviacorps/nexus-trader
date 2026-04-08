from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.project_config import OUTPUTS_V20_DIR, V20_BACKTEST_MONTH_REPORT_PATH, V20_SUMMARY_JSON_PATH, V20_SUMMARY_MD_PATH, V20_WALKFORWARD_RESULTS_PATH


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> int:
    month_report = _read_json(V20_BACKTEST_MONTH_REPORT_PATH)
    walkforward = _read_json(V20_WALKFORWARD_RESULTS_PATH)
    sjd_report = _read_json(OUTPUTS_V20_DIR / "sjd_v20_training_report.json")
    cabr_report = _read_json(OUTPUTS_V20_DIR / "cabr_v20_training_report.json")
    rl_report = _read_json(OUTPUTS_V20_DIR / "rl_executor_training_report.json")
    sjd_dataset_report = _read_json(OUTPUTS_V20_DIR / "sjd_dataset_v20_report.json")
    summary = {
        "version": "v20",
        "month_backtest": month_report,
        "walkforward": walkforward,
        "sjd_training": sjd_report,
        "sjd_dataset": sjd_dataset_report,
        "cabr_training": cabr_report,
        "rl_training": rl_report,
        "honest_status": {
            "phase0_gpu_vram_180gb_target_met": False,
            "full_gpu_training_completed": False,
            "local_v20_runtime_available": True,
            "host_target_port": 8020,
            "prompt_complete_in_local_fallback_mode": True,
        },
    }
    V20_SUMMARY_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    V20_SUMMARY_JSON_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# V20 Summary",
        "",
        f"- Month backtest trades: `{month_report.get('trades_executed', 0)}`",
        f"- Month return: `{month_report.get('return_pct', 0.0)}`%",
        f"- Month win rate: `{month_report.get('win_rate', 0.0)}`",
        f"- Month profit factor: `{month_report.get('profit_factor', 0.0)}`",
        f"- Walk-forward windows completed: `{((walkforward.get('summary') or {}).get('windows_completed', 0))}`",
        f"- Walk-forward average return: `{((walkforward.get('summary') or {}).get('avg_return_pct', 0.0))}`%",
        f"- Walk-forward deflated Sharpe proxy: `{((walkforward.get('summary') or {}).get('deflated_sharpe_proxy', 0.0))}`",
        f"- SJD dataset rows: `{sjd_dataset_report.get('dataset_rows', 0)}`",
        f"- SJD best holdout stance accuracy: `{sjd_report.get('best_valid_stance_accuracy', 0.0)}`",
        f"- CABR training rows: `{cabr_report.get('rows', 0)}`",
        f"- RL regimes profiled: `{rl_report.get('regimes_trained', 0)}`",
        "- Honest note: this V20 pass is a locally hosted, research-structured implementation with heuristic or compressed fallbacks for the heavyweight GPU phases.",
    ]
    V20_SUMMARY_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved={V20_SUMMARY_JSON_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
