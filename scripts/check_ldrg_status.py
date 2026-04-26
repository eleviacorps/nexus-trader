from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.project_config import V14_BACKTRADER_WALKFORWARD_REPORT_PATH, V14_CABR_EVALUATION_REPORT_PATH, V14_LDRG_STATUS_PATH
from src.v14.ldrg import check_ldrg


def main() -> int:
    walk = json.loads(V14_BACKTRADER_WALKFORWARD_REPORT_PATH.read_text(encoding="utf-8"))
    cabr = json.loads(V14_CABR_EVALUATION_REPORT_PATH.read_text(encoding="utf-8"))
    ldrg = check_ldrg(
        cabr_accuracy=float(cabr.get("heldout_pairwise_accuracy_overall", 0.0)),
        wf_win_rate=float(walk.get("aggregate_win_rate", 0.0)),
        wf_profitable_months=float(walk.get("profitable_months", 0)) / max(int(walk.get("month_count", 1)), 1),
        wf_months=int(walk.get("month_count", 0)),
        stage1_stage2_gap=float(walk.get("avg_stage_1_vs_stage_2_gap", 1.0)),
        s3pta_count=int(walk.get("paper_trade_summary", {}).get("count", 0)),
        s3pta_win_rate=walk.get("paper_trade_summary", {}).get("win_rate"),
        rsc_max_calibration_error=walk.get("rsc_summary", {}).get("max_calibration_error"),
        wf_max_monthly_dd=float(walk.get("max_single_month_drawdown_pct", 0.0)) / 100.0,
    )
    payload = {
        "tier": int(ldrg.tier),
        "criteria_met": ldrg.criteria_met,
        "blocking_criteria": ldrg.blocking_criteria,
        "recommendation": ldrg.recommendation,
    }
    V14_LDRG_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    V14_LDRG_STATUS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if ldrg.tier == 1:
        print("==============================================", flush=True)
        print("LDRG TIER 1 COMPLETE", flush=True)
        print("Research phase is done.", flush=True)
        print("Next step: accumulate 200 S3PTA paper trades.", flush=True)
        print("Do not deploy live capital until Tier 2 passes.", flush=True)
        print("==============================================", flush=True)
    print(str(V14_LDRG_STATUS_PATH), flush=True)
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
