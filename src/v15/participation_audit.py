from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_GATE_ORDER = (
    "deployable_regime",
    "lrtd_stability",
    "cpm_predictability",
    "uts_threshold",
    "mbeg_minority",
    "daps_minimum_lot",
)

SKIP_REASON_TO_GATE = {
    "wfri_not_deployable": "deployable_regime",
    "lrtd_suppressed": "lrtd_stability",
    "uts_below_threshold": "uts_threshold",
    "minority_veto": "mbeg_minority",
    "daps_minimum_lot": "daps_minimum_lot",
    "lot_below_minimum": "daps_minimum_lot",
    "cabr_below_minimum": "cpm_predictability",
    "eci_pre_release_avoid": "cpm_predictability",
}


@dataclass
class GateRecord:
    name: str
    passed: int = 0
    blocked: int = 0
    block_reasons: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return int(self.passed + self.blocked)

    @property
    def pass_rate(self) -> float | None:
        total = self.total
        if total <= 0:
            return None
        return float(self.passed / total)


class ParticipationAudit:
    def __init__(self, gate_names: tuple[str, ...] = DEFAULT_GATE_ORDER):
        self.gate_names = tuple(gate_names)
        self.gates = {name: GateRecord(name=name) for name in self.gate_names}
        self.total_bars = 0
        self.final_trades = 0
        self.unmapped_reasons: Counter[str] = Counter()

    def log_bar(self, gate_results: dict[str, tuple[bool, str]]) -> None:
        self.total_bars += 1
        all_passed = True
        for name in self.gate_names:
            if name not in gate_results:
                continue
            passed, reason = gate_results[name]
            if passed:
                self.gates[name].passed += 1
            else:
                self.gates[name].blocked += 1
                if reason:
                    self.gates[name].block_reasons.append(str(reason))
                all_passed = False
        if all_passed:
            self.final_trades += 1

    def report(self) -> dict[str, Any]:
        primary = self.primary_bottleneck()
        return {
            "total_bars": int(self.total_bars),
            "final_trades": int(self.final_trades),
            "final_rate": float(self.final_trades / max(self.total_bars, 1)),
            "primary_bottleneck": primary,
            "gates": {
                name: {
                    "available": gate.total > 0,
                    "passed": int(gate.passed),
                    "blocked": int(gate.blocked),
                    "pass_rate": None if gate.pass_rate is None else float(gate.pass_rate),
                    "top_reasons": _top_reasons(gate.block_reasons, n=5),
                }
                for name, gate in self.gates.items()
            },
            "unmapped_reasons": dict(self.unmapped_reasons),
        }

    def primary_bottleneck(self) -> dict[str, Any] | None:
        available = [
            (name, gate)
            for name, gate in self.gates.items()
            if gate.total > 0
        ]
        if not available:
            return None
        name, gate = max(available, key=lambda item: (item[1].blocked, -(item[1].pass_rate or 1.0)))
        return {
            "gate": name,
            "blocked": int(gate.blocked),
            "pass_rate": None if gate.pass_rate is None else float(gate.pass_rate),
            "top_reasons": _top_reasons(gate.block_reasons, n=5),
        }


def _top_reasons(reasons: list[str], n: int = 5) -> list[list[Any]]:
    return [[reason, int(count)] for reason, count in Counter(reasons).most_common(n)]


def _normalize_skip_reason(skip_reason: str) -> tuple[str | None, str]:
    reason = str(skip_reason or "unknown")
    if reason in SKIP_REASON_TO_GATE:
        return SKIP_REASON_TO_GATE[reason], reason
    lowered = reason.lower()
    if lowered.startswith("pce_not_predictable"):
        return "cpm_predictability", reason
    if lowered.startswith("cpm_") or lowered.startswith("low_agreement") or lowered.startswith("atr_outside") or lowered.startswith("regime_unstable"):
        return "cpm_predictability", reason
    return None, reason


def _gate_results_for_reason(reason: str, gate_names: tuple[str, ...]) -> dict[str, tuple[bool, str]]:
    gate_name, normalized_reason = _normalize_skip_reason(reason)
    if gate_name is None:
        return {}
    gate_results: dict[str, tuple[bool, str]] = {}
    for name in gate_names:
        gate_results[name] = (True, "")
        if name == gate_name:
            gate_results[name] = (False, normalized_reason)
            break
    return gate_results


def load_walkforward_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def audit_walkforward_report(payload: dict[str, Any]) -> dict[str, Any]:
    months = list(payload.get("months", []))
    audit = ParticipationAudit()
    skip_counter: Counter[str] = Counter()
    zero_trade_months: list[str] = []
    month_stats: list[dict[str, Any]] = []

    for month in months:
        month_name = str(month.get("month", "unknown"))
        trades = int(month.get("trades_executed", 0))
        skip_breakdown = {
            str(key): int(value)
            for key, value in dict(month.get("skip_reason_breakdown", {})).items()
        }
        total_candidates = trades + sum(skip_breakdown.values())
        if trades <= 0:
            zero_trade_months.append(month_name)
        month_stats.append(
            {
                "month": month_name,
                "trades": trades,
                "candidates": total_candidates,
                "trade_rate": float(trades / max(total_candidates, 1)),
                "skip_reason_breakdown": skip_breakdown,
            }
        )

        for _ in range(trades):
            audit.log_bar({})

        for reason, count in skip_breakdown.items():
            skip_counter[reason] += int(count)
            gate_results = _gate_results_for_reason(reason, audit.gate_names)
            if not gate_results:
                audit.unmapped_reasons[reason] += int(count)
                continue
            for _ in range(int(count)):
                audit.log_bar(gate_results)

    summary = audit.report()
    summary.update(
        {
            "source_version": payload.get("version"),
            "month_count": int(len(months)),
            "aggregate_trades": int(payload.get("aggregate_trades", summary["final_trades"])),
            "zero_trade_months": zero_trade_months,
            "zero_trade_month_count": int(len(zero_trade_months)),
            "top_skip_reasons": [[reason, int(count)] for reason, count in skip_counter.most_common(10)],
            "months": month_stats,
        }
    )
    return summary
