from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover
    pd = None

try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover
    torch = None

from config.project_config import (  # noqa: E402
    FEATURE_DIM_TOTAL,
    FUSED_FEATURE_MATRIX_PATH,
    LEGACY_PRICE_FEATURES_CSV,
    LEGACY_PRICE_FEATURES_PARQUET,
    MARKET_DYNAMICS_LABELS_PATH,
    MODEL_MANIFEST_PATH,
    OUTPUTS_V8_DIR,
    PRICE_FEATURES_CSV_FALLBACK,
    PRICE_FEATURES_PATH,
    QUANT_FEATURES_CSV_FALLBACK,
    QUANT_FEATURES_PATH,
    V8_ANALOG_CACHE_PATH,
    V8_BRANCH_ARCHIVE_PATH,
    V8_BRANCH_ARCHIVE_REPORT_PATH,
    V8_FAIR_VALUE_FRAME_PATH,
    V8_GARCH_FRAME_PATH,
    V8_HMM_FRAME_PATH,
)
from src.evaluation.walkforward import load_model  # noqa: E402
from src.mcts.analog import get_historical_analog_scorer  # noqa: E402
from src.mcts.tree import expand_binary_tree, iter_leaves  # noqa: E402
from src.pipeline.fusion import load_price_frame, merge_market_dynamics_features  # noqa: E402
from src.quant.hybrid import merge_quant_features  # noqa: E402
from src.simulation.personas import default_personas  # noqa: E402
from src.training.train_tft import autocast_context, split_multihorizon_heads_numpy  # noqa: E402
from src.v8.analog_retrieval import AnalogRetrievalCache, retrieve_analogs  # noqa: E402
from src.v8.branch_selector_v8 import summarize_branch_archive  # noqa: E402


def tagged_path(path: Path, run_tag: str) -> Path:
    if not run_tag:
        return path
    return path.with_name(f"{path.stem}_{run_tag}{path.suffix}")


def _read_frame(path: Path):
    if pd is None:
        raise ImportError("pandas is required to build the v8 branch archive.")
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
    if "timestamp" in frame.columns:
        frame = frame.set_index(pd.to_datetime(frame["timestamp"], errors="coerce")).drop(columns=["timestamp"])
    frame.index = pd.to_datetime(frame.index, errors="coerce")
    return frame.sort_index()


def _resolve_first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(f"No artifact found in: {[str(path) for path in paths]}")


def _resolve_v8_frame(path: Path):
    if path.exists():
        return _read_frame(path)
    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        return _read_frame(csv_path)
    return None


def _load_price_context_frame():
    price_path = _resolve_first_existing(
        [PRICE_FEATURES_PATH, PRICE_FEATURES_CSV_FALLBACK, LEGACY_PRICE_FEATURES_PARQUET, LEGACY_PRICE_FEATURES_CSV]
    )
    frame = load_price_frame(price_path)
    if QUANT_FEATURES_PATH.exists():
        frame = merge_quant_features(frame, _read_frame(QUANT_FEATURES_PATH))
    elif QUANT_FEATURES_CSV_FALLBACK.exists():
        frame = merge_quant_features(frame, _read_frame(QUANT_FEATURES_CSV_FALLBACK))
    if MARKET_DYNAMICS_LABELS_PATH.exists():
        frame = merge_market_dynamics_features(frame, _read_frame(MARKET_DYNAMICS_LABELS_PATH))
    else:
        dynamics_csv = MARKET_DYNAMICS_LABELS_PATH.with_suffix(".csv")
        if dynamics_csv.exists():
            frame = merge_market_dynamics_features(frame, _read_frame(dynamics_csv))
    for extra in [_resolve_v8_frame(V8_HMM_FRAME_PATH), _resolve_v8_frame(V8_GARCH_FRAME_PATH), _resolve_v8_frame(V8_FAIR_VALUE_FRAME_PATH)]:
        if extra is None:
            continue
        for column in extra.columns:
            frame[column] = extra[column].reindex(frame.index).ffill().bfill()
    return frame


def _load_analog_cache(path: Path) -> AnalogRetrievalCache:
    bundle = np.load(path)
    timestamps = np.asarray(bundle["timestamps"], dtype=np.int64).astype("datetime64[ns]")
    return AnalogRetrievalCache(
        features=np.asarray(bundle["features"], dtype=np.float32),
        future_paths=np.asarray(bundle["future_paths"], dtype=np.float32),
        timestamps=timestamps,
        feature_mean=np.asarray(bundle["feature_mean"], dtype=np.float32),
        feature_std=np.asarray(bundle["feature_std"], dtype=np.float32),
        window_size=int(np.asarray(bundle["window_size"]).reshape(-1)[0]),
        sample_stride=int(np.asarray(bundle["sample_stride"]).reshape(-1)[0]),
    )


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return default
        return number
    except Exception:
        return default


def _direction_from_prices(anchor_price: float, predicted_prices: list[float]) -> float:
    final_price = predicted_prices[-1] if predicted_prices else anchor_price
    return 1.0 if final_price >= anchor_price else -1.0


def _alignment_from_bias(bias_value: float, branch_direction: float) -> float:
    bias = max(-1.0, min(1.0, float(bias_value)))
    if abs(bias) < 0.05:
        return 0.5
    target = 1.0 if bias >= 0.0 else -1.0
    return max(0.0, 1.0 - abs(target - branch_direction) / 2.0)


def _regime_match(label: str, branch_direction: float, branch_move_z: float, trend_score: float) -> float:
    regime = str(label or "range")
    abs_move = abs(branch_move_z)
    if regime in {"bullish_trend", "breakout"}:
        return 1.0 if branch_direction > 0 else 0.15
    if regime == "bearish_trend":
        return 1.0 if branch_direction < 0 else 0.15
    if regime == "false_breakout":
        expected = -1.0 if trend_score >= 0.0 else 1.0
        return 1.0 if branch_direction == expected else 0.25
    if regime in {"range", "mean_reversion", "low_volatility_drift"}:
        return float(np.clip(1.0 - max(0.0, abs_move - 1.0) * 0.35, 0.0, 1.0))
    if regime == "panic_news_shock":
        return float(np.clip(0.55 + 0.25 * abs_move, 0.0, 1.0))
    return 0.5


def _cone_containment(paths: np.ndarray, actual_path: np.ndarray) -> tuple[float, float]:
    if paths.size == 0:
        return 0.0, 0.0
    lower = np.min(paths, axis=0)
    upper = np.max(paths, axis=0)
    inside = (actual_path >= lower) & (actual_path <= upper)
    return float(inside.mean()), float(inside.all())


def _build_anchor_positions(frame, *, years: list[int], sample_stride: int, sequence_len: int, horizon_minutes: int, max_samples: int) -> np.ndarray:
    index_years = frame.index.year.to_numpy(dtype=np.int32)
    warmup = max(sequence_len, 64, 24)
    positions = []
    for anchor in range(warmup - 1, len(frame) - horizon_minutes - 1, max(1, sample_stride)):
        if years and int(index_years[anchor]) not in years:
            continue
        positions.append(anchor)
    if max_samples > 0 and len(positions) > max_samples:
        take = np.linspace(0, len(positions) - 1, max_samples, dtype=np.int64)
        positions = [positions[int(idx)] for idx in take.tolist()]
    return np.asarray(positions, dtype=np.int64)


def _predict_state_priors(anchor_positions: np.ndarray, manifest_path: Path) -> dict[int, dict[str, float]]:
    if torch is None or not manifest_path.exists():
        return {}
    fused = np.load(FUSED_FEATURE_MATRIX_PATH, mmap_mode="r")
    model, manifest, device = load_model(manifest_path=manifest_path)
    sequence_len = int(manifest.get("sequence_len", 120))
    horizon_labels = list(manifest.get("horizon_labels", ["5m"]))
    horizon_index = horizon_labels.index("15m") if "15m" in horizon_labels else min(len(horizon_labels) - 1, 0)
    output: dict[int, dict[str, float]] = {}
    usable = [int(pos) for pos in anchor_positions.tolist() if pos >= sequence_len - 1 and pos < len(fused)]
    batch_size = 192
    for start in range(0, len(usable), batch_size):
        batch_positions = usable[start : start + batch_size]
        windows = np.stack([np.asarray(fused[pos - sequence_len + 1 : pos + 1], dtype=np.float32) for pos in batch_positions]).astype(np.float32)
        tensor = torch.from_numpy(windows).to(device)
        with torch.no_grad():
            with autocast_context(device, True, "bfloat16"):
                predictions, diagnostics = model(tensor, return_diagnostics=True)
        prediction_np = predictions.detach().float().cpu().numpy().astype(np.float32)
        direction_prob, hold_prob, confidence_prob = split_multihorizon_heads_numpy(prediction_np, len(horizon_labels))
        regime_probabilities = diagnostics["regime_probabilities"].detach().float().cpu().numpy().astype(np.float32)
        for row_index, anchor in enumerate(batch_positions):
            output[int(anchor)] = {
                "direction_prob_15m": float(direction_prob[row_index, horizon_index]),
                "hold_prob_15m": float(hold_prob[row_index, horizon_index]) if hold_prob.shape[1] > horizon_index else 0.0,
                "confidence_prob_15m": float(confidence_prob[row_index, horizon_index]) if confidence_prob.shape[1] > horizon_index else abs(float(direction_prob[row_index, horizon_index]) - 0.5) * 2.0,
                "route_confidence": float(np.max(regime_probabilities[row_index])) if regime_probabilities.size else 0.0,
            }
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the V8 historical branch archive for selector training.")
    parser.add_argument("--years", default="2024,2025,2026")
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--sample-stride", type=int, default=60)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--alpha", type=float, default=0.65)
    parser.add_argument("--beta", type=float, default=0.35)
    parser.add_argument("--output", default="", help="Optional explicit archive output path.")
    parser.add_argument("--report", default="", help="Optional explicit archive report path.")
    args = parser.parse_args()

    if pd is None:
        raise ImportError("pandas is required to build the v8 branch archive.")
    if not V8_ANALOG_CACHE_PATH.exists():
        raise FileNotFoundError("V8 analog cache is required. Run scripts/build_v8_quant_stack.py first.")

    frame = _load_price_context_frame()
    fused = np.load(FUSED_FEATURE_MATRIX_PATH, mmap_mode="r")
    row_count = min(len(frame), len(fused))
    frame = frame.iloc[:row_count].copy()
    years = [int(part.strip()) for part in str(args.years).split(",") if part.strip()]
    manifest_path = tagged_path(MODEL_MANIFEST_PATH, args.run_tag)
    state_priors = _predict_state_priors(
        _build_anchor_positions(frame, years=years, sample_stride=args.sample_stride, sequence_len=120, horizon_minutes=15, max_samples=args.max_samples),
        manifest_path,
    )
    sequence_len = 120
    if manifest_path.exists():
        try:
            state_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            sequence_len = int(state_manifest.get("sequence_len", sequence_len))
        except Exception:
            pass
    anchor_positions = _build_anchor_positions(
        frame,
        years=years,
        sample_stride=args.sample_stride,
        sequence_len=sequence_len,
        horizon_minutes=15,
        max_samples=args.max_samples,
    )
    analog_cache = _load_analog_cache(V8_ANALOG_CACHE_PATH)
    mcts_analog_scorer = get_historical_analog_scorer()
    personas = default_personas()
    close_values = frame["close"].to_numpy(dtype=np.float32, copy=True)
    open_values = frame["open"].to_numpy(dtype=np.float32, copy=True)
    rows: list[dict[str, Any]] = []

    for sample_id, anchor in enumerate(anchor_positions.tolist()):
        if anchor + 15 >= len(frame):
            continue
        row = frame.iloc[anchor]
        current_price = float(close_values[anchor])
        actual_path = np.asarray(
            [close_values[anchor + 5], close_values[anchor + 10], close_values[anchor + 15]],
            dtype=np.float32,
        )
        entry_open = float(open_values[min(anchor + 1, len(open_values) - 1)])
        exit_close = float(close_values[min(anchor + 15, len(close_values) - 1)])
        current_row = {
            key: float(value)
            for key, value in row.to_dict().items()
            if isinstance(value, (int, float, np.integer, np.floating))
        }
        current_row.setdefault("consensus_score", 0.0)
        trend_score = _safe_float(current_row.get("quant_trend_score"), 0.0)
        state_prior = state_priors.get(anchor, {})
        model_direction_prob = _safe_float(state_prior.get("direction_prob_15m"), 0.5)
        model_hold_prob = _safe_float(state_prior.get("hold_prob_15m"), 0.0)
        model_conf_prob = _safe_float(state_prior.get("confidence_prob_15m"), abs(model_direction_prob - 0.5) * 2.0)
        current_row["consensus_score"] = model_conf_prob

        history_start = max(0, anchor - max(analog_cache.window_size, mcts_analog_scorer.window_size) + 1)
        numeric_history = frame.iloc[history_start : anchor + 1].select_dtypes(include=[np.number]).copy()
        history_rows = numeric_history.fillna(0.0).astype(float).to_dict(orient="records")
        root = expand_binary_tree(
            current_row,
            personas,
            max_depth=args.depth,
            analog_scorer=mcts_analog_scorer,
            history_rows=history_rows[-mcts_analog_scorer.window_size :],
        )
        leaves = iter_leaves(root)
        if not leaves:
            continue
        window_start = max(0, anchor - analog_cache.window_size + 1)
        window_features = np.asarray(fused[window_start : anchor + 1, :36], dtype=np.float32)
        analog_result = retrieve_analogs(analog_cache, window_features, top_k=24)
        weight_sum = max(sum(max(float(leaf.probability_weight), 1e-6) for leaf in leaves), 1e-6)
        sample_rows: list[dict[str, Any]] = []
        dominant_regime = str(row.get("hmm_dominant_regime", "range"))
        expected_vol_15m = max(_safe_float(row.get("v8_expected_vol_15m"), abs(_safe_float(row.get("atr_pct"), 1e-4)) * math.sqrt(3.0)), 1e-6)
        fair_value_dislocation = _safe_float(row.get("v8_fair_value_dislocation"), _safe_float(row.get("quant_kalman_dislocation"), 0.0))
        mean_reversion_pressure = _safe_float(row.get("v8_mean_reversion_pressure"), 0.0)
        news_bias = _safe_float(row.get("news_bias"), 0.0)
        crowd_bias = _safe_float(row.get("crowd_bias"), 0.0)
        macro_bias = _safe_float(row.get("macro_bias"), 0.0)
        vwap_distance = _safe_float(row.get("bb_pct"), 0.0) - 0.5
        atr_normalizer = max(_safe_float(row.get("atr_14"), current_price * expected_vol_15m), 1e-6)
        for branch_index, leaf in enumerate(leaves, start=1):
            predicted_prices = [float(price) for price in leaf.path_prices[:3]]
            while len(predicted_prices) < 3:
                predicted_prices.append(predicted_prices[-1] if predicted_prices else current_price)
            branch_direction = _direction_from_prices(current_price, predicted_prices)
            branch_move_size = (predicted_prices[-1] / max(current_price, 1e-6)) - 1.0
            branch_volatility = float(np.std(np.diff([current_price] + predicted_prices) / max(current_price, 1e-6)))
            branch_move_z = branch_move_size / expected_vol_15m
            volatility_realism = float(np.exp(-max(0.0, abs(branch_move_z) - 1.0) * 0.65))
            analog_similarity = float(analog_result.similarity * (0.65 + 0.35 * (1.0 if np.sign(analog_result.directional_prior or 0.0) == branch_direction else 0.0)))
            analog_disagreement = float(analog_result.disagreement * (1.0 if np.sign(analog_result.directional_prior or 0.0) != branch_direction else 0.75))
            raw_probability = max(float(leaf.probability_weight) / weight_sum, 1e-6)
            generator_probability = float(
                np.clip(
                    (0.55 * raw_probability)
                    + (0.30 * (model_direction_prob if branch_direction > 0 else 1.0 - model_direction_prob))
                    + (0.15 * (1.0 - model_hold_prob)),
                    1e-6,
                    0.999999,
                )
            )
            path_error = float(
                (args.alpha * (abs(predicted_prices[-1] - actual_path[-1]) / max(current_price, 1e-6)))
                + (
                    args.beta
                    * (
                        np.mean(np.abs(np.asarray(predicted_prices, dtype=np.float32) - actual_path))
                        / max(current_price, 1e-6)
                    )
                )
            )
            branch_confidence = float(
                np.clip(
                    0.35 * _safe_float(leaf.branch_fitness, 0.0)
                    + 0.20 * _safe_float(leaf.analog_confidence, 0.0)
                    + 0.20 * model_conf_prob
                    + 0.15 * volatility_realism
                    + 0.10 * (1.0 - min(1.0, abs(fair_value_dislocation) * 40.0)),
                    0.0,
                    1.0,
                )
            )
            sample_rows.append(
                {
                    "sample_id": int(sample_id),
                    "timestamp": str(frame.index[anchor]),
                    "year": int(frame.index[anchor].year),
                    "branch_id": int(branch_index),
                    "dominant_regime": dominant_regime,
                    "generator_probability": generator_probability,
                    "hmm_regime_match": _regime_match(dominant_regime, branch_direction, branch_move_z, trend_score),
                    "hmm_persistence": _safe_float(row.get("hmm_regime_persistence"), _safe_float(row.get("quant_regime_persistence"), 0.0)),
                    "hmm_transition_risk": _safe_float(row.get("hmm_transition_probability"), _safe_float(row.get("quant_transition_risk"), 0.0)),
                    "volatility_realism": volatility_realism,
                    "branch_move_zscore": float(branch_move_z),
                    "fair_value_dislocation": fair_value_dislocation,
                    "mean_reversion_pressure": mean_reversion_pressure,
                    "analog_similarity": analog_similarity,
                    "analog_disagreement": analog_disagreement,
                    "news_consistency": _alignment_from_bias(news_bias, branch_direction),
                    "crowd_consistency": _alignment_from_bias(crowd_bias, branch_direction),
                    "macro_alignment": _alignment_from_bias(macro_bias, branch_direction),
                    "branch_direction": float(branch_direction),
                    "branch_move_size": float(branch_move_size),
                    "branch_volatility": branch_volatility,
                    "vwap_distance": float(vwap_distance + branch_move_size * 4.0),
                    "atr_normalized_move": float(abs(predicted_prices[-1] - current_price) / atr_normalizer),
                    "branch_entropy": float(-generator_probability * math.log(max(generator_probability, 1e-6))),
                    "branch_confidence": branch_confidence,
                    "path_error": path_error,
                    "actual_final_return": float((actual_path[-1] / max(current_price, 1e-6)) - 1.0),
                    "actual_price_5m": float(actual_path[0]),
                    "actual_price_10m": float(actual_path[1]),
                    "actual_price_15m": float(actual_path[2]),
                    "predicted_price_5m": float(predicted_prices[0]),
                    "predicted_price_10m": float(predicted_prices[1]),
                    "predicted_price_15m": float(predicted_prices[2]),
                    "anchor_price": current_price,
                    "entry_open_price": entry_open,
                    "exit_close_price_15m": exit_close,
                    "volatility_scale": float(np.clip(abs(_safe_float(row.get("atr_pct"), expected_vol_15m)) * 100.0, 0.25, 4.0)),
                    "model_direction_prob_15m": model_direction_prob,
                    "model_hold_prob_15m": model_hold_prob,
                    "model_confidence_prob_15m": model_conf_prob,
                    "leaf_branch_fitness": _safe_float(leaf.branch_fitness, 0.0),
                    "leaf_analog_confidence": _safe_float(leaf.analog_confidence, 0.0),
                    "leaf_minority_guardrail": _safe_float(leaf.minority_guardrail, 0.0),
                    "leaf_branch_label": str(leaf.branch_label),
                    "winner_label": 0,
                }
            )
        if not sample_rows:
            continue
        winner_index = int(np.argmin([row_["path_error"] for row_ in sample_rows]))
        sample_rows[winner_index]["winner_label"] = 1
        sample_rows[winner_index]["winning_branch"] = True
        rows.extend(sample_rows)

    archive = pd.DataFrame(rows)
    archive_report = summarize_branch_archive(archive). __dict__ if len(archive) else {
        "samples": 0,
        "branches": 0,
        "winner_positive_rate": 0.0,
        "avg_path_error": 0.0,
    }
    archive_report |= {
        "years": years,
        "sample_stride": int(args.sample_stride),
        "max_samples": int(args.max_samples),
        "depth": int(args.depth),
        "row_count": int(len(frame)),
        "sample_count": int(archive["sample_id"].nunique()) if len(archive) else 0,
        "run_tag": args.run_tag,
    }
    output_path = Path(args.output) if args.output else tagged_path(V8_BRANCH_ARCHIVE_PATH, args.run_tag)
    report_path = Path(args.report) if args.report else tagged_path(V8_BRANCH_ARCHIVE_REPORT_PATH, args.run_tag)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        archive.to_parquet(output_path)
        saved_path = output_path
    except Exception:
        saved_path = output_path.with_suffix(".csv")
        archive.to_csv(saved_path, index=False)
    archive_report["artifact_path"] = str(saved_path)
    report_path.write_text(json.dumps(archive_report, indent=2), encoding="utf-8")
    print(json.dumps(archive_report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
