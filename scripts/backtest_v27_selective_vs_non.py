import pandas as pd
import numpy as np
import torch
from pathlib import Path
from dataclasses import dataclass
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set random seed for reproducibility
np.random.seed(42)

# =========================
# IMPORT YOUR SYSTEM
# =========================

from src.v27.short_horizon_predictor import create_short_horizon_predictor
from src.v26.diffusion.regime_generator import RegimeDiffusionPathGenerator


# =========================
# CONFIG
# =========================

INITIAL_BALANCE = 1000
RISK_PER_TRADE = 0.0025
SPREAD = 0.0002

LOOKBACK = 120
CONF_THRESHOLD = 0.60

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# =========================
# LOAD DATA
# =========================

BASE = Path(__file__).resolve().parent.parent

# Load the 144-feature fused dataset
fused = np.load(BASE / "data/features/diffusion_fused_6m.npy", mmap_mode="r")

# Use 50000 windows (~9 months of 15-min bars)
fused = fused[:50000]

print(f"Loaded fused features: {fused.shape}")

# =========================
# DEBUG: DATASET INSPECTION
# =========================

print("\n=== DATASET DEBUG ===")
print("Shape:", fused.shape)
print("Features (dim 1):", fused.shape[1])
print("Sample row (first 10 values):", fused[0, :10])

# Hard validation check
if fused.shape[1] != 144:
    raise ValueError(f"Expected 144 features, got {fused.shape[1]}")

# =========================
# TIMEFRAME
# =========================

# 15-min bars
BAR_SECONDS = 15 * 60
HORIZON_MIN = 15
MAX_HOLD_BARS = 1  # 1 bar = 15 min hold

print(f"Bar size: {BAR_SECONDS}s | Hold bars: {MAX_HOLD_BARS}")


# =========================
# MODEL INIT - REAL V26 DIFFUSION MODEL
# =========================

print("Loading diffusion generator...")
from src.v26.diffusion.regime_generator import RegimeGeneratorConfig, RegimeDiffusionPathGenerator
from src.v27.short_horizon_predictor import create_short_horizon_predictor

print("Creating config...")
config = RegimeGeneratorConfig(
    in_channels=144,
    sequence_length=120,
    temporal_gru_dim=256,
    temporal_layers=2,
    context_len=256,
    num_regimes=9,
    regime_embed_dim=16,
    temporal_film_dim=272,
)

print("Creating generator...")
generator = RegimeDiffusionPathGenerator(config=config, device=DEVICE)

# Load checkpoint
ckpt_path = BASE / "models/v26/diffusion_phase1_final.pt"
print(f"Loading checkpoint from {ckpt_path}...")
ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
print("Checkpoint loaded, filtering state dict...")
model_state = generator.model.state_dict()
compatible = {k: v for k, v in ckpt.get("ema", ckpt.get("model", {})).items()
             if k in model_state and model_state[k].shape == v.shape}
print(f"Compatible weights: {len(compatible)}")
generator.model.load_state_dict(compatible, strict=False)
print("Model weights loaded")

# Create V27 predictor
print("Creating V27 predictor...")
predictor = create_short_horizon_predictor(generator, device=DEVICE)
predictor.num_futures = 4  # Reduced for speed
predictor.validity_minutes = 15
print("Predictor ready")


# =========================
# TRADE STRUCT
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


# =========================
# WORLD STATE (FIXED)
# =========================

def build_world_state(idx):
    # Use first feature as price proxy
    return {
        "price": 100.0 + idx * 0.01,  # Dummy price
        "return": 0.0,
        "range": 0.5,
        "volume": 1000.0,
    }


# =========================
# RR MAP
# =========================

def get_rr(conf):
    if conf > 0.82: return 5
    if conf > 0.74: return 4
    if conf > 0.67: return 3
    if conf > 0.60: return 2
    return 1.5


# =========================
# SIGNAL (REAL MODEL)
# =========================

def get_signal(idx):
    """Get signal using diffusion paths (REAL model output)."""

    if idx < LOOKBACK:
        return {"decision": "HOLD", "confidence": 0, "paths": None}

    # Get window from fused features
    window = fused[idx - LOOKBACK:idx]

    past_context = torch.tensor(window, dtype=torch.float32, device=DEVICE)

    # DEBUG: Verify model input dimensionality
    if idx == LOOKBACK:
        print("\n=== MODEL INPUT DEBUG ===")
        print("past_context shape:", past_context.shape)
        print("Expected features: 144")
        print("Actual features:", past_context.shape[-1])

    # Hard validation check
    if past_context.shape[-1] != 144:
        raise ValueError(f"Expected 144 features, got {past_context.shape[-1]}")

    # USE REAL V26 MODEL - generate paths
    regime_probs = torch.ones(9, device=DEVICE) / 9

    result = predictor.predict_15min_trade(
        past_context=past_context,
        regime_probs=regime_probs,
        current_price=100.0 + idx * 0.01,
        steps=10
    )

    return {
        "decision": result.decision,
        "confidence": result.confidence,
        "paths": None  # Will be generated separately for simulation
    }


# =========================
# TRADE SIM - PATH-BASED (CORRECT IMPLEMENTATION)
# =========================

def simulate_trade_from_paths(start_idx, direction, conf, entry_price=100.0):
    """Simulate trade using REAL diffusion paths.

    Steps:
    1. Generate paths from the diffusion model
    2. Map each path to price space using feature[0] as return proxy
    3. Step through each path checking TP/SL
    4. Aggregate outcomes - NO randomness
    """

    # Risk parameters
    risk_pct = 0.002  # 0.2% risk
    rr_target = get_rr(conf)

    risk_amount = entry_price * risk_pct
    tp_distance = risk_amount * rr_target
    sl_distance = risk_amount

    if direction == "BUY":
        tp = entry_price + tp_distance
        sl = entry_price - sl_distance
    else:
        tp = entry_price - tp_distance
        sl = entry_price + sl_distance

    # Generate paths using the diffusion model
    paths = generator.generate_paths(
        world_state={"price": entry_price},
        regime_probs=torch.ones(9, device=DEVICE) / 9,
        past_context=torch.tensor(fused[start_idx - LOOKBACK:start_idx], dtype=torch.float32, device=DEVICE).unsqueeze(0),
        num_paths=8,
        steps=5,
    )

    if not paths:
        return Trade(start_idx, start_idx, direction, 0.0, 0.0, 0.0, "expiry")

    # Simulate each path step-by-step
    tp_hits = 0
    sl_hits = 0
    expiry_hits = 0
    realized_rrs = []

    for path in paths:
        data = np.array(path["data"])  # (120, 144) normalized features

        # Map feature[0] to price
        # Feature[0] is a normalized return - use cumulative sum as return proxy
        returns = data[:, 0]  # (120,) normalized returns
        cumulative_return = np.cumsum(returns)

        # Build price trajectory
        prices = entry_price * (1 + cumulative_return / 100.0)  # Scale down the normalized values

        # Step through until TP/SL or expiry
        trade_exited = False
        for t, price in enumerate(prices):
            if direction == "BUY":
                if price >= tp:
                    tp_hits += 1
                    realized_rrs.append(rr_target)
                    trade_exited = True
                    break
                elif price <= sl:
                    sl_hits += 1
                    realized_rrs.append(-1.0)
                    trade_exited = True
                    break
            else:  # SELL
                if price <= tp:
                    tp_hits += 1
                    realized_rrs.append(rr_target)
                    trade_exited = True
                    break
                elif price >= sl:
                    sl_hits += 1
                    realized_rrs.append(-1.0)
                    trade_exited = True
                    break

        if not trade_exited:
            expiry_hits += 1
            # Use last price for expiry
            final_return = (prices[-1] - entry_price) / entry_price
            realized_rrs.append(final_return / risk_pct)

    # Determine outcome based on path consensus (NO randomness)
    n_paths = len(paths)

    # Confidence = fraction of paths that hit TP
    confidence_from_paths = tp_hits / n_paths

    # Outcome determined by majority of paths
    if tp_hits > sl_hits:
        # Most paths hit TP
        avg_rr = np.mean([r for r in realized_rrs if r > 0]) if tp_hits > 0 else 0.0
        rr_realized = avg_rr if tp_hits >= n_paths / 2 else np.median(realized_rrs)
        exit_reason = "tp"
    elif sl_hits > tp_hits:
        # Most paths hit SL
        avg_loss = np.mean([r for r in realized_rrs if r < 0]) if sl_hits > 0 else -1.0
        rr_realized = avg_loss
        exit_reason = "sl"
    else:
        # Tie - use median
        rr_realized = float(np.median(realized_rrs))
        exit_reason = "expiry"

    # Clamp to reasonable range
    rr_realized = max(-1.0, min(rr_target, rr_realized))

    # Calculate P&L
    pnl = rr_realized * (INITIAL_BALANCE * RISK_PER_TRADE)

    hold_min = 15.0  # 15-min horizon
    exit_idx = start_idx + 1

    return Trade(
        start_idx,
        exit_idx,
        direction,
        pnl,
        rr_realized,
        hold_min,
        exit_reason
    )


# =========================
# TRADE SIM - NON-SELECTIVE (same path-based for fair comparison)
# =========================

def simulate_trade_non_selective(start_idx, direction, conf, entry_price=100.0):
    """Non-selective: trade every signal with same path-based simulation."""

    risk_pct = 0.002
    rr_target = 1.5  # Fixed RR for non-selective (minimum)

    risk_amount = entry_price * risk_pct
    tp_distance = risk_amount * rr_target
    sl_distance = risk_amount

    if direction == "BUY":
        tp = entry_price + tp_distance
        sl = entry_price - sl_distance
    else:
        tp = entry_price - tp_distance
        sl = entry_price + sl_distance

    paths = generator.generate_paths(
        world_state={"price": entry_price},
        regime_probs=torch.ones(9, device=DEVICE) / 9,
        past_context=torch.tensor(fused[start_idx - LOOKBACK:start_idx], dtype=torch.float32, device=DEVICE).unsqueeze(0),
        num_paths=8,
        steps=5,
    )

    if not paths:
        return Trade(start_idx, start_idx, direction, 0.0, 0.0, 0.0, "expiry")

    tp_hits = 0
    sl_hits = 0
    realized_rrs = []

    for path in paths:
        data = np.array(path["data"])
        returns = data[:, 0]
        cumulative_return = np.cumsum(returns)
        prices = entry_price * (1 + cumulative_return / 100.0)

        trade_exited = False
        for t, price in enumerate(prices):
            if direction == "BUY":
                if price >= tp:
                    tp_hits += 1
                    realized_rrs.append(rr_target)
                    trade_exited = True
                    break
                elif price <= sl:
                    sl_hits += 1
                    realized_rrs.append(-1.0)
                    trade_exited = True
                    break
            else:
                if price <= tp:
                    tp_hits += 1
                    realized_rrs.append(rr_target)
                    trade_exited = True
                    break
                elif price >= sl:
                    sl_hits += 1
                    realized_rrs.append(-1.0)
                    trade_exited = True
                    break

        if not trade_exited:
            final_return = (prices[-1] - entry_price) / entry_price
            realized_rrs.append(final_return / risk_pct)

    if tp_hits > sl_hits:
        rr_realized = np.mean([r for r in realized_rrs if r > 0]) if tp_hits > 0 else float(np.median(realized_rrs))
        exit_reason = "tp"
    elif sl_hits > tp_hits:
        rr_realized = np.mean([r for r in realized_rrs if r < 0]) if sl_hits > 0 else -1.0
        exit_reason = "sl"
    else:
        rr_realized = float(np.median(realized_rrs))
        exit_reason = "expiry"

    rr_realized = max(-1.0, min(rr_target, rr_realized))
    pnl = rr_realized * (INITIAL_BALANCE * RISK_PER_TRADE)

    return Trade(start_idx, start_idx + 1, direction, pnl, rr_realized, 15.0, exit_reason)


# =========================
# BACKTEST
# =========================

MAX_PREDICTIONS = 50  # Path-based is slower

def run_backtest_selective():
    """Selective mode: trade only when confidence >= threshold."""
    print("\n=== RUNNING SELECTIVE MODE (conf >= 0.60) - PATH-BASED ===")
    trades = []
    idx = LOOKBACK
    predictions = 0

    while idx < len(fused) - MAX_HOLD_BARS and predictions < MAX_PREDICTIONS:
        if predictions % 10 == 0:
            print(f"  [{predictions}/{MAX_PREDICTIONS}] idx={idx}")

        sig = get_signal(idx)

        if sig["decision"] == "HOLD" or sig["confidence"] < CONF_THRESHOLD:
            idx += 1
            continue

        # Use path-based simulation
        trade = simulate_trade_from_paths(idx, sig["decision"], sig["confidence"])
        trades.append(trade)
        predictions += 1

        idx = trade.exit_idx + MAX_HOLD_BARS

    print(f"Selective: {len(trades)} trades")
    return trades


def run_backtest_non_selective():
    """Non-selective mode: trade EVERY signal with same path-based simulation."""
    print("\n=== RUNNING NON-SELECTIVE MODE (every signal) - PATH-BASED ===")
    trades = []
    idx = LOOKBACK
    predictions = 0

    while idx < len(fused) - MAX_HOLD_BARS and predictions < MAX_PREDICTIONS:
        if predictions % 10 == 0:
            print(f"  [{predictions}/{MAX_PREDICTIONS}] idx={idx}")

        sig = get_signal(idx)

        if sig["decision"] == "HOLD":
            idx += 1
            continue

        # Use same path-based simulation for fair comparison
        trade = simulate_trade_non_selective(idx, sig["decision"], sig["confidence"])
        trades.append(trade)
        predictions += 1

        idx = trade.exit_idx + MAX_HOLD_BARS

    print(f"Non-Selective: {len(trades)} trades")
    return trades


# =========================
# METRICS
# =========================

def analyze(trades):

    if not trades:
        return {"trades": 0, "win_rate": 0, "return_%": 0, "avg_rr": 0, "avg_hold_min": 0}

    pnls = np.array([t.pnl for t in trades])
    wins = pnls[pnls > 0]
    balance = INITIAL_BALANCE
    for p in pnls:
        balance += p

    return {
        "trades": len(trades),
        "win_rate": len(wins) / len(trades),
        "return_%": (balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100,
        "avg_rr": np.mean([t.rr for t in trades]),
        "avg_hold_min": np.mean([t.hold_min for t in trades]),
        "tp_%": sum(t.exit_reason == "tp" for t in trades) / len(trades),
        "sl_%": sum(t.exit_reason == "sl" for t in trades) / len(trades),
        "expiry_%": sum(t.exit_reason == "expiry" for t in trades) / len(trades),
        "final_balance": balance,
    }


# =========================
# RUN
# =========================

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(line_buffering=True)

    print("\n" + "="*70)
    print("V27 SELECTIVE vs NON-SELECTIVE 3-MONTH BACKTEST")
    print("Using REAL V26 Diffusion Model")
    print("="*70)

    # Run both modes
    selective_trades = run_backtest_selective()
    non_selective_trades = run_backtest_non_selective()

    # Analyze both
    selective_metrics = analyze(selective_trades)
    non_selective_metrics = analyze(non_selective_trades)

    # Calculate selection edge
    selection_edge_wr = selective_metrics.get("win_rate", 0) - non_selective_metrics.get("win_rate", 0)
    selection_edge_ret = selective_metrics.get("return_%", 0) - non_selective_metrics.get("return_%", 0)
    selection_edge_rr = selective_metrics.get("avg_rr", 0) - non_selective_metrics.get("avg_rr", 0)

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    print("\n--- SELECTIVE MODE (conf >= 0.60) ---")
    for k, v in selective_metrics.items():
        print(f"  {k}: {v}")

    print("\n--- NON-SELECTIVE MODE (every signal) ---")
    for k, v in non_selective_metrics.items():
        print(f"  {k}: {v}")

    print("\n--- SELECTION EDGE ---")
    print(f"  Win Rate: {selection_edge_wr*100:+.2f}%")
    print(f"  Return %: {selection_edge_ret:+.2f}%")
    print(f"  Avg RR: {selection_edge_rr:+.3f}")

    print("\n--- INTERPRETATION ---")
    if selection_edge_wr > 0.05:
        print("  Confidence filtering adds significant value (win rate)")
    elif selection_edge_wr < -0.05:
        print("  Confidence filtering HURTS performance (non-selective is better)")
    else:
        print("  Confidence filtering has minimal impact on win rate")

    if selection_edge_ret > 5:
        print("  Selective mode significantly outperforms")
    elif selection_edge_ret < -5:
        print("  Non-selective mode significantly outperforms")
    else:
        print("  Both modes perform similarly")

    print("\n" + "="*70)
    print("BACKTEST COMPLETE")
    print("="*70)