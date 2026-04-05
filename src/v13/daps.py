from __future__ import annotations

REGIME_SCALARS = {
    'trending_up': 1.00,
    'trending_down': 0.90,
    'breakout': 0.80,
    'ranging': 0.60,
    'panic_shock': 0.30,
    'low_volatility': 0.50,
    'unknown': 0.40,
}


def daps_lot_size(
    base_capital: float,
    current_equity: float,
    recent_win_rate: float,
    regime: str,
    uts_score: float,
    stop_pips: float = 20.0,
    pip_value_per_lot: float = 10.0,
) -> float:
    base_risk_pct = 0.01
    performance_scalar = max(0.3, min(1.5, float(recent_win_rate) / 0.55))
    regime_scalar = float(REGIME_SCALARS.get(str(regime), REGIME_SCALARS['unknown']))
    conviction_scalar = 0.7 + 0.6 * float(uts_score)
    risk_pct = base_risk_pct * performance_scalar * regime_scalar * conviction_scalar
    risk_amount = float(current_equity) * min(risk_pct, 0.025)
    lot_size = risk_amount / max(float(stop_pips) * float(pip_value_per_lot), 1e-6)
    return round(max(0.01, min(float(lot_size), 1.0)), 2)
