from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
BACKTRADER_ROOT = PROJECT_ROOT / "SimilarExistingSolutions" / "backtrader-master"
if str(BACKTRADER_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKTRADER_ROOT))

import backtrader as bt  # type: ignore

from config.project_config import (
    FUSED_FEATURE_MATRIX_PATH,
    FUSED_TIMESTAMPS_PATH,
    GATE_CONTEXT_PATH,
    MODEL_MANIFEST_PATH,
    TARGETS_MULTIHORIZON_PATH,
    TFT_MODEL_DIR,
)
from src.data.fused_dataset import DatasetSlice
from src.evaluation.walkforward import (
    _combined_gate_scores,
    apply_bucket_calibration,
    load_model,
    predict_multihorizon_for_slice,
    resolve_meta_gate_path,
    resolve_precision_gate_path,
)
from src.training.meta_gate import load_meta_gate


class NexusSignalData(bt.feeds.PandasData):
    lines = ("signal",)
    params = (("signal", -1),)


class NexusSignalStrategy(bt.Strategy):
    params = dict(hold_bars=15, stake=0.01)

    def __init__(self) -> None:
        self.signal_line = self.datas[0].signal
        self.entry_bar: int | None = None
        self.closed_pnls: list[float] = []
        self.closed_pnls_comm: list[float] = []
        self.order_count = 0

    def next(self) -> None:
        if self.position:
            if self.entry_bar is not None and len(self) - self.entry_bar >= int(self.p.hold_bars):
                self.close()
            return

        signal = float(self.signal_line[0])
        if signal > 0:
            self.buy(size=float(self.p.stake))
            self.order_count += 1
            self.entry_bar = len(self)
        elif signal < 0:
            self.sell(size=float(self.p.stake))
            self.order_count += 1
            self.entry_bar = len(self)

    def notify_trade(self, trade) -> None:
        if trade.isclosed:
            self.closed_pnls.append(float(trade.pnl))
            self.closed_pnls_comm.append(float(trade.pnlcomm))
            self.entry_bar = None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_signal_frame(
    run_tag: str,
    horizon: str,
    *,
    use_external_gate: bool = True,
    gate_threshold_override: float | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_path = TFT_MODEL_DIR / f"model_manifest_{run_tag}.json"
    walkforward_path = PROJECT_ROOT / "outputs" / "evaluation" / f"walkforward_report_{run_tag}.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    if not walkforward_path.exists():
        raise FileNotFoundError(f"Missing walkforward report: {walkforward_path}")

    model, manifest, device = load_model(manifest_path=manifest_path)
    sequence_len = int(manifest.get("sequence_len", 120))
    horizon_labels = list(manifest.get("horizon_labels", ["5m", "10m", "15m", "30m"]))
    if horizon not in horizon_labels:
        raise ValueError(f"Horizon {horizon} not found in {horizon_labels}")
    horizon_idx = horizon_labels.index(horizon)

    timestamps = np.load(FUSED_TIMESTAMPS_PATH, mmap_mode="r")
    feature_count = np.load(FUSED_FEATURE_MATRIX_PATH, mmap_mode="r").shape[0]
    usable = feature_count - sequence_len
    row_slice = DatasetSlice(0, usable)

    targets, probabilities = predict_multihorizon_for_slice(
        model,
        device,
        row_slice,
        feature_path=FUSED_FEATURE_MATRIX_PATH,
        target_bundle_path=TARGETS_MULTIHORIZON_PATH,
        target_keys=list(manifest.get("output_labels", [])),
        sequence_len=sequence_len,
        batch_size=1024,
        amp_enabled=bool(manifest.get("amp_enabled", False)),
        amp_dtype=str(manifest.get("amp_dtype", "bfloat16")),
    )
    direction_targets, hold_targets, confidence_targets = np.split(targets, 3, axis=1)
    direction_probabilities, hold_probabilities, confidence_probabilities = np.split(probabilities, 3, axis=1)

    walkforward = _load_json(walkforward_path)
    calibration = walkforward.get("overall", {}).get("calibration", {})
    thresholds = walkforward.get("optimized_thresholds", {})
    decision_threshold = float(thresholds.get("decision_threshold", 0.53))
    confidence_floor = float(thresholds.get("confidence_floor", 0.06))
    hold_threshold = float(thresholds.get("hold_threshold", 0.55))
    gate_threshold = float(thresholds.get("gate_threshold", 0.5))
    if gate_threshold_override is not None:
        gate_threshold = float(gate_threshold_override)

    gate_scores = None
    if use_external_gate:
        precision_gate_path = resolve_precision_gate_path(manifest)
        precision_gate = _load_json(precision_gate_path) if precision_gate_path is not None and precision_gate_path.exists() else None
        meta_gate_path = resolve_meta_gate_path(manifest)
        try:
            meta_gate = load_meta_gate(meta_gate_path) if meta_gate_path is not None else None
        except Exception:
            meta_gate = None
        context_features = None
        if GATE_CONTEXT_PATH.exists():
            context_arr = np.load(GATE_CONTEXT_PATH, mmap_mode="r")
            context_features = np.asarray(context_arr[sequence_len - 1 : sequence_len - 1 + usable], dtype=np.float32)

        _, _, gate_scores = _combined_gate_scores(
            probabilities,
            precision_gate,
            meta_gate,
            context_features=context_features,
        )

    horizon_probabilities = apply_bucket_calibration(direction_probabilities[:, horizon_idx], calibration)
    horizon_hold = hold_targets[:, horizon_idx]
    horizon_hold_prob = hold_probabilities[:, horizon_idx]
    horizon_conf_prob = confidence_probabilities[:, horizon_idx]

    price = pd.read_csv(PROJECT_ROOT / "data" / "features" / "price_features.csv")
    price["datetime"] = pd.to_datetime(price["datetime"], utc=True)
    price = price.iloc[: feature_count].copy()
    signal = np.zeros(len(price), dtype=np.float32)

    start_idx = sequence_len - 1
    end_idx = start_idx + usable
    for local_idx, price_idx in enumerate(range(start_idx, min(end_idx, len(price) - 1))):
        prob = float(horizon_probabilities[local_idx])
        conf = float(horizon_conf_prob[local_idx])
        hold_prob = float(horizon_hold_prob[local_idx])
        gate_ok = True if gate_scores is None else float(gate_scores[local_idx]) >= gate_threshold
        direction = 0.0
        if hold_prob >= hold_threshold or conf < confidence_floor or not gate_ok:
            direction = 0.0
        elif prob >= decision_threshold:
            direction = 1.0
        elif prob <= (1.0 - decision_threshold):
            direction = -1.0
        signal[price_idx] = direction

    price["signal"] = signal
    price = price[["datetime", "open", "high", "low", "close", "volume", "signal"]].copy()
    price = price.set_index("datetime")
    meta = {
        "run_tag": run_tag,
        "horizon": horizon,
        "sequence_len": sequence_len,
        "decision_threshold": decision_threshold,
        "confidence_floor": confidence_floor,
        "hold_threshold": hold_threshold,
        "gate_threshold": gate_threshold,
        "gate_threshold_override": None if gate_threshold_override is None else float(gate_threshold_override),
        "use_external_gate": bool(use_external_gate),
        "signal_count": int(np.count_nonzero(signal)),
        "positive_targets_mean": float(direction_targets[:, horizon_idx].mean()),
        "hold_targets_mean": float(horizon_hold.mean()),
    }
    return price, meta


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a one-week Backtrader backtest from local V8 signals.")
    parser.add_argument("--run-tag", default="mh12_recent_v8")
    parser.add_argument("--horizon", default="15m")
    parser.add_argument("--capital", type=float, default=100.0)
    parser.add_argument("--stake", type=float, default=0.01)
    parser.add_argument("--hold-bars", type=int, default=15)
    parser.add_argument("--commission-bps", type=float, default=2.0)
    parser.add_argument("--slippage-perc", type=float, default=0.0002)
    parser.add_argument("--disable-external-gate", action="store_true")
    parser.add_argument("--gate-threshold-override", type=float, default=None)
    args = parser.parse_args()

    frame, meta = _build_signal_frame(
        args.run_tag,
        args.horizon,
        use_external_gate=not args.disable_external_gate,
        gate_threshold_override=args.gate_threshold_override,
    )
    data = NexusSignalData(dataname=frame)

    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.setcash(float(args.capital))
    cerebro.broker.setcommission(commission=float(args.commission_bps) / 10000.0)
    cerebro.broker.set_slippage_perc(float(args.slippage_perc), slip_open=True, slip_limit=True, slip_match=True, slip_out=False)
    cerebro.adddata(data)
    cerebro.addstrategy(NexusSignalStrategy, hold_bars=int(args.hold_bars), stake=float(args.stake))

    start_value = float(cerebro.broker.getvalue())
    results = cerebro.run()
    strategy = results[0]
    end_value = float(cerebro.broker.getvalue())
    trade_count = len(strategy.closed_pnls_comm)
    wins = sum(1 for pnl in strategy.closed_pnls_comm if pnl > 0)
    losses = sum(1 for pnl in strategy.closed_pnls_comm if pnl < 0)

    report = {
        "run_tag": args.run_tag,
        "horizon": args.horizon,
        "start": str(frame.index.min()),
        "end": str(frame.index.max()),
        "bars": int(len(frame)),
        "signal_count": int((frame["signal"] != 0).sum()),
        "trade_count": int(trade_count),
        "wins": int(wins),
        "losses": int(losses),
        "win_rate": round(float(wins / trade_count), 6) if trade_count else 0.0,
        "gross_pnl_sum": round(float(np.sum(strategy.closed_pnls)), 6) if strategy.closed_pnls else 0.0,
        "net_pnl_sum": round(float(np.sum(strategy.closed_pnls_comm)), 6) if strategy.closed_pnls_comm else 0.0,
        "avg_net_pnl": round(float(np.mean(strategy.closed_pnls_comm)), 6) if strategy.closed_pnls_comm else 0.0,
        "start_cash": round(start_value, 6),
        "final_cash": round(end_value, 6),
        "return_pct": round(((end_value / start_value) - 1.0) * 100.0, 6) if start_value else 0.0,
        "stake": float(args.stake),
        "hold_bars": int(args.hold_bars),
        "commission_bps": float(args.commission_bps),
        "slippage_perc": float(args.slippage_perc),
        "meta": meta,
    }

    out_path = PROJECT_ROOT / "outputs" / "evaluation" / f"backtrader_week_{args.run_tag}_{args.horizon}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(str(out_path))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
