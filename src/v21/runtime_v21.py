from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class RuntimeDecision:
    should_trade: bool
    failed_gates: list[str]
    mode: str


class V21Runtime:
    """
    V21 runtime with explicit research vs production gating.

    Research mode measures raw signal quality by trading every non-HOLD stance.
    Production mode applies the V21 safety stack before execution.
    """

    def __init__(self, mode: str = "production") -> None:
        mode_value = str(mode).strip().lower()
        if mode_value not in {"research", "production"}:
            raise ValueError(f"Invalid mode: {mode}")
        self.mode = mode_value

    def should_trade(
        self,
        sjd_output: Mapping[str, Any],
        conformal_confidence: float,
        dangerous_branch_count: int,
        meta_label_prob: float,
    ) -> tuple[bool, list[str]]:
        stance = str(sjd_output.get("stance", "HOLD")).upper()
        if self.mode == "research":
            return (stance != "HOLD"), ([] if stance != "HOLD" else ["stance"])

        disagree_prob = float(sjd_output.get("disagree_prob", 1.0))
        gates = {
            "conformal": float(conformal_confidence) >= 0.55,
            "dangerous": int(dangerous_branch_count) <= 2,
            "disagreement": disagree_prob <= 0.65,
            "meta_label": float(meta_label_prob) >= 0.40,
            "stance": stance != "HOLD",
        }
        failed = [name for name, passed in gates.items() if not passed]
        return (len(failed) == 0), failed

    def decide(
        self,
        sjd_output: Mapping[str, Any],
        conformal_confidence: float,
        dangerous_branch_count: int,
        meta_label_prob: float,
    ) -> RuntimeDecision:
        should_trade, failed = self.should_trade(
            sjd_output=sjd_output,
            conformal_confidence=conformal_confidence,
            dangerous_branch_count=dangerous_branch_count,
            meta_label_prob=meta_label_prob,
        )
        return RuntimeDecision(should_trade=should_trade, failed_gates=failed, mode=self.mode)

    def get_size(self, kelly_fraction: float, account_balance: float, price: float) -> float:
        max_risk_pct = 0.02
        max_lot = 0.20
        price_value = max(float(price), 1e-6)
        kelly_lot = max(float(kelly_fraction), 0.0) * float(account_balance) / price_value
        risk_lot = (float(account_balance) * max_risk_pct) / (price_value * 100.0)
        return round(max(0.0, min(kelly_lot, risk_lot, max_lot)), 6)


__all__ = ["RuntimeDecision", "V21Runtime"]
