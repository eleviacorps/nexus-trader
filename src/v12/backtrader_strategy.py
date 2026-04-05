from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import backtrader as bt  # type: ignore
except Exception:  # pragma: no cover
    bt = None


@dataclass(frozen=True)
class V12PlannedTrade:
    sample_id: int
    decision_ts: str
    exit_signal_ts: str
    direction: int
    planned_lot: float
    score: float
    stage_name: str


@dataclass(frozen=True)
class V13SkipDecision:
    sample_id: int
    decision_ts: str
    reason: str
    regime: str
    uts_score: float


def scaled_lot_for_equity(
    equity: float,
    *,
    start_lot: float,
    max_lot: float,
    start_equity: float,
    end_equity: float,
) -> float:
    if equity <= start_equity:
        return float(start_lot)
    if equity >= end_equity:
        return float(max_lot)
    progress = (equity - start_equity) / max(end_equity - start_equity, 1e-6)
    return float(start_lot + progress * (max_lot - start_lot))


def build_v12_signal_plan(
    frame,
    *,
    score_column: str,
    threshold: float,
    contract_size_oz: float = 100.0,
    lot_for_equity=None,
) -> list[dict[str, Any]]:
    working = frame.copy()
    ranked = working.sort_values(["sample_id", score_column], ascending=[True, False], kind="mergesort").groupby("sample_id", sort=False).head(1).copy()
    ranked = ranked.loc[ranked[score_column] >= float(threshold)].copy()
    timestamps = ranked["timestamp"].astype(str)
    stage_bars = ranked["stage_bars"].to_numpy(dtype=np.int32)
    decision_ts = (np.asarray(timestamps, dtype="datetime64[ns]") + stage_bars.astype("timedelta64[m]")).astype("datetime64[ns]")
    exit_ts = (np.asarray(timestamps, dtype="datetime64[ns]") + np.full(len(ranked), 15, dtype=np.int32).astype("timedelta64[m]")).astype("datetime64[ns]")
    plans: list[dict[str, Any]] = []
    for row, decision, exit_signal in zip(ranked.to_dict(orient="records"), decision_ts, exit_ts, strict=False):
        lot = float(row.get("planned_lot", 0.1))
        if callable(lot_for_equity):
            try:
                lot = float(lot_for_equity(float(row.get("reference_equity", 0.0))))
            except Exception:
                lot = float(row.get("planned_lot", 0.1))
        plans.append(
            {
                "sample_id": int(row["sample_id"]),
                "decision_ts": str(decision),
                "exit_signal_ts": str(exit_signal),
                "direction": int(np.sign(float(row.get("setl_trade_direction", 1.0)) or 1.0)),
                "planned_lot": lot,
                "planned_size_oz": float(lot * contract_size_oz),
                "score": float(row[score_column]),
                "confidence": float(row.get("calibrated_confidence", row.get("confidence", 0.5))),
                "stage_name": str(row.get("stage_name", "open")),
                "dominant_regime": str(row.get("dominant_regime", "ranging")),
            }
        )
    return plans


if bt is not None:
    class V12SignalData(bt.feeds.PandasData):
        params = (("datetime", None), ("open", -1), ("high", -1), ("low", -1), ("close", -1), ("volume", -1), ("openinterest", None))


    class NexusV12Strategy(bt.Strategy):
        params = dict(
            plan_map=None,
            planned_skips=None,
            start_lot=0.10,
            max_lot=1.0,
            start_equity=1000.0,
            end_equity=2500.0,
            contract_size_oz=100.0,
        )

        def __init__(self) -> None:
            self.plan_map = dict(self.p.plan_map or {})
            self.skipped_log: list[dict[str, Any]] = list(self.p.planned_skips or [])
            self.pending_order = None
            self.active_plan = None
            self.trades_log: list[dict[str, Any]] = []
            self.equity_curve: list[dict[str, Any]] = []

        def next(self) -> None:
            dt = bt.num2date(self.datas[0].datetime[0]).replace(tzinfo=None)
            self.equity_curve.append({"datetime": dt, "equity": float(self.broker.getvalue())})
            if self.pending_order is not None:
                return
            if self.position and self.active_plan is not None and dt >= self.active_plan["exit_signal_ts"]:
                self.pending_order = self.close()
                return
            if (not self.position) and dt in self.plan_map:
                self.active_plan = dict(self.plan_map[dt])
                lot = float(self.active_plan.get("planned_lot", 0.0) or 0.0)
                if lot <= 0.0:
                    lot = scaled_lot_for_equity(
                        float(self.broker.getvalue()),
                        start_lot=float(self.p.start_lot),
                        max_lot=float(self.p.max_lot),
                        start_equity=float(self.p.start_equity),
                        end_equity=float(self.p.end_equity),
                    )
                size = float(lot * float(self.p.contract_size_oz))
                self.active_plan["planned_lot"] = lot
                self.active_plan["planned_size_oz"] = size
                self.active_plan["entry_dt"] = dt
                if int(self.active_plan.get("direction", 1)) > 0:
                    self.pending_order = self.buy(size=size)
                else:
                    self.pending_order = self.sell(size=size)

        def notify_order(self, order) -> None:
            if order.status in [order.Submitted, order.Accepted]:
                return
            if order.status == order.Completed and self.active_plan is not None:
                if self.position:
                    self.active_plan["entry_fill_price"] = float(order.executed.price)
                else:
                    self.active_plan["exit_fill_price"] = float(order.executed.price)
            self.pending_order = None

        def notify_trade(self, trade) -> None:
            if trade.isclosed and self.active_plan is not None:
                record = dict(self.active_plan)
                record["gross_pnl"] = float(trade.pnl)
                record["net_pnl"] = float(trade.pnlcomm)
                record["exit_dt"] = bt.num2date(self.datas[0].datetime[0]).replace(tzinfo=None)
                self.trades_log.append(record)
                self.active_plan = None
