"""Metrics for adjuster quality evaluation."""

from __future__ import annotations

import numpy as np


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def direction_accuracy(path: np.ndarray, target: np.ndarray) -> float:
    p = np.sign(np.diff(path))
    t = np.sign(np.diff(target))
    return float((p == t).mean())


def smoothness(path: np.ndarray) -> tuple[float, float]:
    v = np.diff(path)
    a = np.diff(v)
    return float(np.mean(np.abs(v))), float(np.mean(np.abs(a)))


def evaluate_adjustment(
    selected_path: np.ndarray,
    refined_path: np.ndarray,
    target_future: np.ndarray,
) -> dict[str, float]:
    mse_before = mse(selected_path, target_future)
    mse_after = mse(refined_path, target_future)
    improvement = 100.0 * (mse_before - mse_after) / max(mse_before, 1e-8)

    vol_error = float(abs(np.std(refined_path) - np.std(target_future)))
    dir_acc = direction_accuracy(refined_path, target_future) * 100.0

    max_refined = float(np.max(np.abs(refined_path)))
    max_target = float(np.max(np.abs(target_future)))
    max_move_ratio = max_refined / max(max_target, 1e-8)

    vel_norm, accel_norm = smoothness(refined_path)

    return {
        "mse_before": mse_before,
        "mse_after": mse_after,
        "improvement_pct": improvement,
        "volatility_error": vol_error,
        "direction_acc_pct": dir_acc,
        "max_move_ratio": max_move_ratio,
        "velocity_norm": vel_norm,
        "accel_norm": accel_norm,
    }


def print_metrics(result: dict[str, float]) -> None:
    print("=== Adjuster Metrics ===")
    print(f"mse_before={result['mse_before']:.6f}")
    print(f"mse_after={result['mse_after']:.6f}")
    print(f"improvement_pct={result['improvement_pct']:.2f}%")
    print(f"volatility_error={result['volatility_error']:.6f}")
    print(f"direction_acc_pct={result['direction_acc_pct']:.2f}%")
    print(f"max_move_ratio={result['max_move_ratio']:.4f}")
    print(f"velocity_norm={result['velocity_norm']:.6f}")
    print(f"accel_norm={result['accel_norm']:.6f}")
