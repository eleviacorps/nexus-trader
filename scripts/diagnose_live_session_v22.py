from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.v22.live_autopsy import build_live_session_diagnostic, write_live_session_diagnostic


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the V22 Phase 0 live-session autopsy.")
    parser.add_argument("--date", default="2026-04-10")
    parser.add_argument("--source-path", default="")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--magic", type=int, default=21042026)
    args = parser.parse_args()

    payload = build_live_session_diagnostic(
        session_date=str(args.date),
        source_path=str(args.source_path).strip() or None,
        symbol=str(args.symbol).upper(),
        magic=int(args.magic),
    )
    out_path = write_live_session_diagnostic(payload, session_date=str(args.date))
    print(str(out_path))
    print(
        json.dumps(
            {
                "session_date": payload.get("session_date"),
                "trade_count": payload.get("trade_count"),
                "wins": payload.get("wins"),
                "losses": payload.get("losses"),
                "net_profit": payload.get("net_profit"),
                "win_rate": payload.get("win_rate"),
                "realized_rr": payload.get("realized_rr"),
                "max_consecutive_losses": (payload.get("loss_streak") or {}).get("max_consecutive_losses"),
                "sell_trades": ((payload.get("by_direction") or {}).get("SELL") or {}).get("trades"),
                "pretrade_rr_below_1_5_count": payload.get("pretrade_rr_below_1_5_count"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
