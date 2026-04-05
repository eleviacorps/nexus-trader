from __future__ import annotations

from typing import Optional

import numpy as np


def branch_survival_score(
    branch_prices: np.ndarray,
    current_atr: float,
    n_perturbations: int = 30,
    perturbation_scale: float = 0.30,
    seed: Optional[int] = None,
) -> float:
    prices = np.asarray(branch_prices, dtype=np.float32)
    atr = max(float(current_atr), 0.0)
    if prices.size < 2:
        return 0.5
    if atr <= 0.0:
        return 1.0 if prices[-1] != prices[0] else 0.5
    rng = np.random.default_rng(seed)
    original_direction = 1 if prices[-1] > prices[0] else -1
    noise_std = atr * float(perturbation_scale)
    survival_count = 0
    for _ in range(max(int(n_perturbations), 1)):
        noise = rng.normal(0.0, noise_std, prices.size)
        perturbed = prices + noise
        perturbed_direction = 1 if perturbed[-1] > perturbed[0] else -1
        if perturbed_direction == original_direction:
            survival_count += 1
    return float(survival_count / max(int(n_perturbations), 1))


def batch_branch_survival(
    all_branch_prices: list[np.ndarray],
    current_atr: float,
    n_perturbations: int = 30,
) -> np.ndarray:
    return np.asarray(
        [
            branch_survival_score(prices, current_atr=current_atr, n_perturbations=n_perturbations)
            for prices in all_branch_prices
        ],
        dtype=np.float32,
    )
