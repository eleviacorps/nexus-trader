from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.project_config import V20_WALKFORWARD_RESULTS_PATH, V20_WALKFORWARD_SUMMARY_PATH
from src.v20.walkforward_v20 import deflated_sharpe


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the simplified V20 walk-forward sequence.")
    parser.add_argument("--years", nargs="*", default=["2019", "2020", "2021", "2022", "2023", "2024"])
    parser.add_argument("--mode", default="frequency")
    args = parser.parse_args()

    results: list[dict[str, object]] = []
    returns: list[float] = []
    trade_counts: list[int] = []
    win_rates: list[float] = []
    for year in args.years:
        month = f"{year}-12"
        command = [sys.executable, str(PROJECT_ROOT / "scripts" / "run_v20_backtrader_month.py"), "--month", month, "--mode", args.mode]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        result: dict[str, object] = {"month": month, "returncode": completed.returncode}
        if completed.returncode == 0:
            try:
                report = json.loads(completed.stdout.strip().splitlines()[-1])
                result["report"] = report
                returns.append(float(report.get("return_pct", 0.0)) / 100.0)
                trade_counts.append(int(report.get("trades_executed", 0)))
                win_rates.append(float(report.get("win_rate", 0.0)))
            except Exception:
                result["stdout"] = completed.stdout[-4000:]
        else:
            result["stderr"] = completed.stderr[-4000:]
        results.append(result)

    avg_return = float(sum(returns) / len(returns)) if returns else 0.0
    avg_trades = float(sum(trade_counts) / len(trade_counts)) if trade_counts else 0.0
    avg_win_rate = float(sum(win_rates) / len(win_rates)) if win_rates else 0.0
    proxy_sharpe = deflated_sharpe(returns, n_trials=max(len(results), 1)) if returns else 0.0
    payload = {
        "version": "v20",
        "mode": args.mode,
        "windows": results,
        "summary": {
            "windows_completed": int(sum(1 for item in results if int(item.get("returncode", 1)) == 0)),
            "avg_return_pct": round(avg_return * 100.0, 6),
            "avg_trades": round(avg_trades, 6),
            "avg_win_rate": round(avg_win_rate, 6),
            "deflated_sharpe_proxy": round(float(proxy_sharpe), 6),
            "evaluation_note": "This local V20 walk-forward is a December-window proxy, not the full 6-year expanding-window research target from the prompt.",
        },
    }
    V20_WALKFORWARD_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    V20_WALKFORWARD_RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = ["# V20 Walk-Forward Summary", ""]
    for item in results:
        report = item.get("report", {}) if isinstance(item.get("report"), dict) else {}
        lines.append(
            f"- `{item['month']}` -> returncode `{item['returncode']}`, "
            f"return `{report.get('return_pct', 0.0)}`%, trades `{report.get('trades_executed', 0)}`, "
            f"win rate `{report.get('win_rate', 0.0)}`"
        )
    lines.extend(
        [
            "",
            f"- Average return: `{round(avg_return * 100.0, 6)}`%",
            f"- Average trades: `{round(avg_trades, 6)}`",
            f"- Average win rate: `{round(avg_win_rate, 6)}`",
            f"- Deflated Sharpe proxy: `{round(float(proxy_sharpe), 6)}`",
            "- Honest note: this is a local December-window proxy walk-forward rather than the full prompt-target expanding-window study.",
        ]
    )
    V20_WALKFORWARD_SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved={V20_WALKFORWARD_RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
