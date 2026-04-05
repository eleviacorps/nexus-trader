from __future__ import annotations

import numpy as np

REGIME_SCALARS = {
    'trending_up': 1.00,
    'trending_down': 0.90,
    'breakout': 0.80,
    'ranging': 0.60,
    'panic_shock': 0.30,
    'low_volatility': 0.50,
    'unknown': 0.40,
}


def maximum_lot_for_leverage(
    *,
    current_equity: float,
    max_account_leverage: float | None,
    price_per_ounce: float | None,
    contract_size_oz: float = 100.0,
) -> float | None:
    if max_account_leverage is None or float(max_account_leverage) <= 0.0:
        return None
    if price_per_ounce is None or float(price_per_ounce) <= 0.0:
        return None
    notional_cap = float(current_equity) * float(max_account_leverage)
    per_lot_notional = float(price_per_ounce) * float(contract_size_oz)
    if per_lot_notional <= 0.0:
        return None
    return float(max(notional_cap / per_lot_notional, 0.0))


def daps_lot_size(
    base_capital: float,
    current_equity: float,
    recent_win_rate: float,
    regime: str,
    uts_score: float,
    stop_pips: float = 20.0,
    pip_value_per_lot: float = 10.0,
    *,
    base_risk_pct: float = 0.01,
    hard_cap_risk_pct: float = 0.025,
    min_lot: float = 0.05,
    max_lot: float = 1.00,
    max_account_leverage: float | None = None,
    price_per_ounce: float | None = None,
    contract_size_oz: float = 100.0,
) -> float:
    performance_scalar = max(0.3, min(1.5, float(recent_win_rate) / 0.55))
    regime_scalar = float(REGIME_SCALARS.get(str(regime), REGIME_SCALARS['unknown']))
    conviction_scalar = 0.7 + 0.6 * float(uts_score)
    risk_pct = float(base_risk_pct) * performance_scalar * regime_scalar * conviction_scalar
    risk_amount = float(current_equity) * min(risk_pct, float(hard_cap_risk_pct))
    lot_size = risk_amount / max(float(stop_pips) * float(pip_value_per_lot), 1e-6)
    leverage_cap = maximum_lot_for_leverage(
        current_equity=float(current_equity),
        max_account_leverage=max_account_leverage,
        price_per_ounce=price_per_ounce,
        contract_size_oz=contract_size_oz,
    )
    if leverage_cap is not None:
        lot_size = min(float(lot_size), float(leverage_cap))
        if float(leverage_cap) + 1e-9 < float(min_lot):
            return 0.0
    clipped = float(np.clip(lot_size, 0.0, float(max_lot)))
    if clipped <= 0.0:
        return 0.0
    return round(float(np.clip(clipped, float(min_lot), float(max_lot))), 2)
