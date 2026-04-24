"""V27 Selective vs Non-Selective 3-Month Backtest - HYBRID MODEL (CORRECTED).

Architecture:
- Diffusion model → decision + confidence + RR
- Real OHLC data → TP/SL/Expiry execution
- Perfect timestamp alignment between fused features and OHLCV
- NO randomness in trade outcomes
"""

import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

np.random.seed(42)

# =========================
# CONFIG
# =========================
INITIAL_BALANCE = 1000.0
RISK_PER_TRADE = 0.0025
SPREAD = 0.0002
LOOKBACK = 120
CONF_THRESHOLD = 0.50
HORIZON_MIN = 60  # 1 hour
BAR_SECONDS = 15 * 60
ATR_PERIOD = 14
ATR_K = 2.0  # SL = 2.0 ATR (wider, more realistic)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Exit mode: 'fixed_rr' = TP/SL targets, 'hold' = exit at horizon
EXIT_MODE = "hold"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# LOAD DATA
# =========================
BASE = Path(__file__).resolve().parent.parent

# OHLCV (15-min bars) - filtered to 3 months
ohlcv = pd.read_parquet(BASE / "data/features/v21_ohlcv_denoised.parquet")
ohlcv.index = pd.to_datetime(ohlcv.index).tz_convert('UTC')

START_DATE = "2024-01-01"
END_DATE = "2024-03-31"
ohlcv = ohlcv[START_DATE:END_DATE].copy()
# Keep datetime index before reset
ohlcv_dts = ohlcv.index.copy()
ohlcv = ohlcv.reset_index(drop=True)

print(f"OHLCV: {len(ohlcv)} bars ({START_DATE} to {END_DATE})")
print(f"Date range: {ohlcv_dts[0]} to {ohlcv_dts[-1]}")

# Fused features (1-min, for model input)
fused = np.load(BASE / "data/features/diffusion_fused_6m.npy", mmap_mode="r")
ts_fused = np.load(BASE / "data/features/diffusion_timestamps_6m.npy", mmap_mode='r')
ts_fused = pd.to_datetime(ts_fused).tz_localize('UTC')
fused_index = pd.Index(ts_fused)

# Build alignment: which fused index corresponds to each OHLCV bar
ts_fused_pd = pd.Index(pd.to_datetime(ts_fused))

# Vectorized searchsorted using saved datetime index
alignment = ts_fused_pd.searchsorted(ohlcv_dts)

# Clip to valid range
alignment = np.clip(alignment, 0, len(ts_fused_pd) - 1)

print(f"Fused: {fused.shape}, Alignment range: {alignment.min()} to {alignment.max()}")

# Verify alignment
min_fused = alignment.min()
if min_fused < LOOKBACK:
    print(f"WARNING: First {LOOKBACK - min_fused} bars need more lookback, skipping")
    skip_bars = LOOKBACK - min_fused
else:
    skip_bars = 0


# =========================
# MODEL INIT
# =========================
print("\nLoading V26 diffusion model...")

from src.v26.diffusion.regime_generator import RegimeGeneratorConfig, RegimeDiffusionPathGenerator
from src.v27.short_horizon_predictor import create_short_horizon_predictor

config = RegimeGeneratorConfig(
    in_channels=144, sequence_length=120, temporal_gru_dim=256,
    temporal_layers=2, context_len=256, num_regimes=9,
    regime_embed_dim=16, temporal_film_dim=272,
)

generator = RegimeDiffusionPathGenerator(config=config, device=DEVICE)

ckpt = torch.load(BASE / "models/v26/diffusion_phase1_final.pt", map_location=DEVICE, weights_only=False)
model_state = generator.model.state_dict()
compatible = {k: v for k, v in ckpt.get("ema", ckpt.get("model", {})).items()
               if k in model_state and model_state[k].shape == v.shape}
generator.model.load_state_dict(compatible, strict=False)
print(f"Loaded {len(compatible)} weights")

predictor = create_short_horizon_predictor(generator, device=DEVICE)
predictor.confidence_threshold = 0.35  # Lower to see real model behavior
predictor.num_futures = 16  # More samples


# =========================
# TRADE STRUCTURE
# =========================
@dataclass
class Trade:
    entry_idx: int
    exit_idx: int
    direction: str
    pnl: float
    rr: float
    hold_min: float
    exit_reason: str
    confidence: float


# =========================
# RR MAPPING
# =========================
def get_rr(conf):
    # Standard 15-min RR: 1.5-2
    return 1.5 + conf * 0.5


# =========================
# ATR COMPUTATION
# =========================
def compute_atr(ohlcv_df, idx, period=14):
    """Compute ATR at given index using prior bars."""
    if idx < 1:
        return 1.0  # fallback

    trs = []
    for i in range(idx - period + 1, idx):
        if i < 0:
            continue
        high = ohlcv_df.iloc[i]["high"]
        low = ohlcv_df.iloc[i]["low"]
        prev_close = ohlcv_df.iloc[i - 1]["close"] if i > 0 else ohlcv_df.iloc[i]["close"]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
    return np.mean(trs) if trs else 1.0


# =========================
# GET SIGNAL FROM MODEL
# =========================
def get_signal(ohlcv_idx):
    """Get signal from diffusion model using aligned indices."""
    fused_idx = alignment[ohlcv_idx]

    if fused_idx < LOOKBACK:
        return {"decision": "HOLD", "confidence": 0.0}

    window = fused[fused_idx - LOOKBACK:fused_idx]
    past_context = torch.tensor(window, dtype=torch.float32, device=DEVICE)

    regime_probs = torch.ones(9, device=DEVICE) / 9

    result = predictor.predict_15min_trade(
        past_context=past_context,
        regime_probs=regime_probs,
        current_price=float(ohlcv.iloc[ohlcv_idx]["close"]),
        steps=5
    )

    return {
        "decision": result.decision,
        "confidence": result.confidence
    }


# =========================
# SIMULATE TRADE (REAL OHLC EXECUTION)
# =========================
def simulate_trade(ohlcv_idx, direction, conf, balance):
    """Simulate trade - supports both fixed RR (TP/SL) and hold-to-horizon."""

    entry_raw = float(ohlcv.iloc[ohlcv_idx]["close"])
    rr_target = get_rr(conf)
    atr = compute_atr(ohlcv, ohlcv_idx, ATR_PERIOD)

    entry = entry_raw + SPREAD if direction == "BUY" else entry_raw - SPREAD
    sl_distance = atr * ATR_K
    tp_distance = sl_distance * rr_target
    max_bars = max(1, int(HORIZON_MIN * 60 / BAR_SECONDS))

    print(f"    Entry: {entry_raw:.2f}, dir={direction}, conf={conf:.2f}, rr={rr_target}, atr={atr:.2f}")

    if EXIT_MODE == "hold":
        # Simple hold-to-horizon: exit at bar close after N bars
        exit_idx = min(ohlcv_idx + max_bars, len(ohlcv) - 1)
        exit_raw = float(ohlcv.iloc[exit_idx]["close"])
        exit_price = exit_raw - SPREAD if direction == "BUY" else exit_raw + SPREAD
        exit_reason = "expiry"
    else:
        # Fixed RR: TP/SL with trailing check
        if direction == "BUY":
            tp = entry_raw + tp_distance
            sl = entry_raw - sl_distance
        else:
            tp = entry_raw - tp_distance
            sl = entry_raw + sl_distance

        exit_reason = "expiry"
        exit_price = entry
        exit_idx = ohlcv_idx

        for offset in range(1, max_bars + 1):
            check_idx = ohlcv_idx + offset
            if check_idx >= len(ohlcv):
                break

            bar = ohlcv.iloc[check_idx]
            high = bar["high"]
            low = bar["low"]

            if direction == "BUY":
                if high >= tp:
                    exit_reason = "tp"
                    exit_price = tp - SPREAD
                    exit_idx = check_idx
                    break
                if low <= sl:
                    exit_reason = "sl"
                    exit_price = sl + SPREAD
                    exit_idx = check_idx
                    break
            else:
                if low <= tp:
                    exit_reason = "tp"
                    exit_price = tp + SPREAD
                    exit_idx = check_idx
                    break
                if high >= sl:
                    exit_reason = "sl"
                    exit_price = sl - SPREAD
                    exit_idx = check_idx
                    break

        if exit_reason == "expiry":
            exit_raw = float(ohlcv.iloc[exit_idx]["close"])
            exit_price = exit_raw - SPREAD if direction == "BUY" else exit_raw + SPREAD

    print(f"      exit={exit_reason}, price={exit_price:.2f}")

    # Correct RR: realized / 1R
    if direction == "BUY":
        realized_r = (exit_price - entry) / sl_distance
    else:
        realized_r = (entry - exit_price) / sl_distance

    pnl = realized_r * (balance * RISK_PER_TRADE)

    hold_min = (exit_idx - ohlcv_idx) * 15 / 60

    return Trade(
        entry_idx=ohlcv_idx,
        exit_idx=exit_idx,
        direction=direction,
        pnl=pnl,
        rr=realized_r,
        hold_min=hold_min,
        exit_reason=exit_reason,
        confidence=conf
    )


# =========================
# BACKTEST
# =========================
MAX_TRADES = 50

def run_selective(non_overlap=True):
    """Trade only when confidence >= 0.60."""
    print(f"\n=== SELECTIVE MODE (conf >= 0.60, {'non-overlap' if non_overlap else 'overlap'}) ===")
    trades = []
    idx = skip_bars
    count = 0
    balance = INITIAL_BALANCE

    while idx < len(ohlcv) - 1 and count < MAX_TRADES:
        if count % 20 == 0:
            print(f"  [{count}/{MAX_TRADES}] idx={idx}")

        sig = get_signal(idx)

        if sig["decision"] == "HOLD" or sig["confidence"] < CONF_THRESHOLD:
            idx += 1
            continue

        trade = simulate_trade(idx, sig["decision"], sig["confidence"], balance)
        trades.append(trade)
        balance += trade.pnl
        count += 1
        idx = trade.exit_idx + 1 if non_overlap else idx + 1

    print(f"Selective: {len(trades)} trades")
    return trades


def run_non_selective(non_overlap=True):
    """Trade every BUY/SELL signal."""
    print(f"\n=== NON-SELECTIVE MODE (every signal, {'non-overlap' if non_overlap else 'overlap'}) ===")
    trades = []
    idx = skip_bars
    count = 0
    balance = INITIAL_BALANCE

    while idx < len(ohlcv) - 1 and count < MAX_TRADES:
        if count % 20 == 0:
            print(f"  [{count}/{MAX_TRADES}] idx={idx}")

        sig = get_signal(idx)

        if sig["decision"] == "HOLD":
            idx += 1
            continue

        trade = simulate_trade(idx, sig["decision"], sig["confidence"], balance)
        trades.append(trade)
        balance += trade.pnl
        count += 1
        idx = trade.exit_idx + 1 if non_overlap else idx + 1

    print(f"Non-selective: {len(trades)} trades")
    return trades


# =========================
# METRICS
# =========================
def analyze(trades):
    if not trades:
        return {"trades": 0}

    pnls = np.array([t.pnl for t in trades])
    rrs = np.array([t.rr for t in trades])
    wins = pnls[pnls > 0]

    balance = INITIAL_BALANCE
    for p in pnls:
        balance += p

    peak = INITIAL_BALANCE
    max_dd = 0
    running = INITIAL_BALANCE
    for p in pnls:
        running += p
        if running > peak:
            peak = running
        dd = (peak - running) / peak
        if dd > max_dd:
            max_dd = dd

    return {
        "trades": len(trades),
        "win_rate": len(wins) / len(trades),
        "return_pct": (balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100,
        "avg_rr": float(np.mean(rrs)),
        "avg_hold_min": float(np.mean([t.hold_min for t in trades])),
        "tp_pct": sum(t.exit_reason == "tp" for t in trades) / len(trades),
        "sl_pct": sum(t.exit_reason == "sl" for t in trades) / len(trades),
        "expiry_pct": sum(t.exit_reason == "expiry" for t in trades) / len(trades),
        "max_drawdown_pct": max_dd * 100,
        "final_balance": balance,
        "avg_confidence": float(np.mean([t.confidence for t in trades])),
    }


# =========================
# RUN
# =========================
def get_signal_stats(n=500):
    """Analyze signal distribution."""
    from collections import Counter
    decisions = Counter()
    confidences = []
    
    for i in range(skip_bars, skip_bars + n):
        sig = get_signal(i)
        decisions[sig["decision"]] += 1
        confidences.append(sig["confidence"])
    
    return decisions, confidences


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)

    print("\n" + "=" * 70)
    print("V27 DIRECTIONAL ACCURACY TEST")
    print("=" * 70)

    # Test directional accuracy (1-hour horizon, 100 samples)
    correct = 0
    total = 0
    for i in range(skip_bars, min(skip_bars + 100, len(ohlcv) - 4)):
        sig = get_signal(i)
        if sig["decision"] == "HOLD":
            continue
        curr = float(ohlcv.iloc[i]["close"])
        future = float(ohlcv.iloc[i + 4]["close"])
        if (sig["decision"] == "BUY" and future > curr) or (sig["decision"] == "SELL" and future < curr):
            correct += 1
        total += 1

    dir_acc = correct / total if total > 0 else 0
    print(f"\n  Directional Accuracy (1h horizon): {correct}/{total} = {dir_acc:.1%}")
    print(f"  Baseline (random): 50%")

    # Signal distribution check
    print("\n=== SIGNAL DISTRIBUTION (first 100 bars) ===")
    decisions, confidences = get_signal_stats(100)
    print(f"  Decisions: {dict(decisions)}")
    print(f"  Confidence: min={min(confidences):.2f}, max={max(confidences):.2f}, mean={np.mean(confidences):.2f}")

    # Non-overlapping (current, conservative)
    selective_trades = run_selective(non_overlap=True)
    non_selective_trades = run_non_selective(non_overlap=True)

    sel = analyze(selective_trades)
    non = analyze(non_selective_trades)

    print("\n" + "=" * 70)
    print("RESULTS (non-overlapping)")
    print("=" * 70)

    print("\n--- SELECTIVE MODE ---")
    for k, v in sel.items():
        print(f"  {k}: {v}")

    print("\n--- NON-SELECTIVE MODE ---")
    for k, v in non.items():
        print(f"  {k}: {v}")

    print("\n--- SELECTION EDGE ---")
    print(f"  Win Rate: {(sel['win_rate'] - non['win_rate']) * 100:+.2f}%")
    print(f"  Return: {sel['return_pct'] - non['return_pct']:+.2f}%")
    print(f"  Avg RR: {sel['avg_rr'] - non['avg_rr']:+.3f}")
    print(f"  TP%: {(sel['tp_pct'] - non['tp_pct']) * 100:+.2f}%")

    print("\n" + "=" * 70)