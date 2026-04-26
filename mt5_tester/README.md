# V21 MT5 Strategy Tester Bridge

This bridge lets MetaTrader 5 Strategy Tester replay exported V21 signals.

Important:
- MT5 Strategy Tester is not running the live Python V21 stack directly.
- The Python side exports timestamped V21 BUY/SELL decisions, lots, and SL/TP values into a CSV.
- The included Expert Advisor reads that CSV and replays those signals inside MT5 Strategy Tester.

## Files

- EA: [NexusTraderV21TesterBridge.mq5](/c:/PersonalDrive/Programming/AiStudio/nexus-trader/mt5_tester/NexusTraderV21TesterBridge.mq5)
- exporter: [export_v21_mt5_tester_bridge.py](/c:/PersonalDrive/Programming/AiStudio/nexus-trader/scripts/export_v21_mt5_tester_bridge.py)

## Export V21 Signals

Run from the repo root:

```powershell
C:\Users\rfsga\miniconda3\python.exe scripts\export_v21_mt5_tester_bridge.py --month 2023-12 --mode precision --equity 10000 --copy-to-mt5-common
```

That produces:

- CSV: `outputs/v21/mt5_tester/v21_mt5_tester_signals.csv`
- summary: `outputs/v21/mt5_tester/v21_mt5_tester_summary.json`

With `--copy-to-mt5-common`, the CSV is also copied into MT5 Common Files so the tester EA can read it without a manual file copy.

## Install The EA

1. Copy [NexusTraderV21TesterBridge.mq5](/c:/PersonalDrive/Programming/AiStudio/nexus-trader/mt5_tester/NexusTraderV21TesterBridge.mq5) into your MT5 `MQL5/Experts` folder.
2. Open MetaEditor and compile it.
3. Open MT5 Strategy Tester.
4. Choose:
   - Expert: `NexusTraderV21TesterBridge`
   - Symbol: `XAUUSD` or your broker-specific mapped symbol
   - Period: `M15`
5. In the EA inputs:
   - `SignalCsvFile = v21_mt5_tester_signals.csv`
   - `UseCommonFiles = true` if you used `--copy-to-mt5-common`
   - optionally set `LotOverride` if you want tester-side fixed size instead of the exported V21 lot

## What Gets Replayed

Each CSV row contains:

- `execution_time_utc`
- `action`
- `lot`
- `stop_loss`
- `take_profit`
- V21 metadata such as `confidence_tier`, `cabr_score`, `cpm_score`, and `branch_label`

The EA opens the matching market order on the specified `M15` execution bar. If an opposite signal appears later, it can close the current position first and then open the new one.

## Limits

- This is a replay bridge, not a full Python-in-MT5 runtime.
- News, Kimi, and external live feeds are not being recomputed inside Strategy Tester.
- Results can differ from the native Python V21 backtest because MT5 fill rules, spread modeling, and tester execution semantics are different.
