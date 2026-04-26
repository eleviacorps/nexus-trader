from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from config.project_config import (
    V17_KIMI_PACKET_LOG_PATH,
    V18_KIMI_PACKET_LOG_PATH,
    V19_SJD_DATASET_PATH,
    V19_SJD_DATASET_REPORT_PATH,
    V19_SJD_FEATURE_NAMES_PATH,
    V19_SJD_TEACHER_CACHE_PATH,
)
from src.v19.context_sampler import SimulationContextSampler, feature_map_from_context, load_context_source_frame

TEACHER_MODELS: tuple[str, ...] = (
    "moonshotai/kimi-k2-5",
    "nvidia/llama-3.3-nemotron-super-49b-v1",
    "meta/llama-3.1-70b-instruct",
)

SUPPLEMENTAL_COLUMNS: tuple[str, ...] = (
    "generator_probability",
    "analog_similarity",
    "analog_disagreement",
    "news_consistency",
    "crowd_consistency",
    "macro_alignment",
    "branch_confidence",
    "path_error",
    "actual_final_return",
    "model_direction_prob_15m",
    "model_hold_prob_15m",
    "model_confidence_prob_15m",
    "leaf_analog_confidence",
    "v10_diversity_score",
    "hmm_transition_risk",
    "volatility_realism",
    "fair_value_dislocation",
    "macro_bias",
    "macro_shock",
    "news_bias",
    "news_intensity",
    "crowd_bias",
    "crowd_extreme",
    "consensus_score",
    "retail_impact",
    "institutional_impact",
    "algo_impact",
    "whale_impact",
    "quant_regime_strength",
    "quant_transition_risk",
    "quant_vol_realism",
    "quant_fair_value_z",
    "quant_route_confidence",
    "quant_trend_score",
    "hurst_overall",
    "hurst_positive",
    "hurst_negative",
    "hurst_asymmetry",
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        if np.isnan(number) or np.isinf(number):
            return float(default)
        return number
    except Exception:
        return float(default)


def _packet_log_paths(paths: Iterable[Path] | None = None) -> list[Path]:
    return [path for path in list(paths or [V17_KIMI_PACKET_LOG_PATH, V18_KIMI_PACKET_LOG_PATH]) if Path(path).exists()]


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            yield payload


def _timestamp_from_row(row: Mapping[str, Any]) -> pd.Timestamp:
    candles = ((row.get("context") or {}).get("market") or {}).get("candles", [])
    candle_timestamp = candles[-1].get("timestamp") if isinstance(candles, list) and candles else None
    candidates = [
        row.get("packet_bucket_15m_utc"),
        row.get("logged_at"),
        candle_timestamp,
    ]
    for candidate in candidates:
        timestamp = pd.to_datetime(candidate, utc=True, errors="coerce")
        if pd.notna(timestamp):
            return timestamp
    return pd.Timestamp.now(tz="UTC")


def _response_offsets(content: Mapping[str, Any], current_price: float, pip_size: float = 0.1) -> tuple[float, float, float]:
    entry_zone = content.get("entry_zone", [])
    if isinstance(entry_zone, list) and len(entry_zone) == 2:
        entry_center = (_safe_float(entry_zone[0], current_price) + _safe_float(entry_zone[1], current_price)) / 2.0
    else:
        entry_center = current_price
    stop_loss = _safe_float(content.get("stop_loss"), current_price)
    take_profit = _safe_float(content.get("take_profit"), current_price)
    return (
        (entry_center - current_price) / max(float(pip_size), 1e-9),
        (stop_loss - current_price) / max(float(pip_size), 1e-9),
        (take_profit - current_price) / max(float(pip_size), 1e-9),
    )


@dataclass
class HistoricalJoiner:
    frame: pd.DataFrame

    @classmethod
    def load(cls) -> "HistoricalJoiner":
        frame = load_context_source_frame().copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame = frame.sort_values("timestamp").reset_index(drop=True)
        return cls(frame=frame)

    def lookup(self, timestamp: pd.Timestamp) -> dict[str, float]:
        if self.frame.empty:
            return {}
        ts = pd.to_datetime(timestamp, utc=True, errors="coerce")
        if pd.isna(ts):
            return {}
        idx = self.frame["timestamp"].searchsorted(ts, side="left")
        candidates = [max(min(idx, len(self.frame) - 1), 0)]
        if idx > 0:
            candidates.append(idx - 1)
        chosen = min(
            candidates,
            key=lambda item: abs((self.frame.iloc[item]["timestamp"] - ts).total_seconds()),
        )
        row = self.frame.iloc[int(chosen)]
        output: dict[str, float] = {}
        for column in SUPPLEMENTAL_COLUMNS:
            if column not in row.index:
                continue
            output[f"archive.{column}"] = _safe_float(row[column], 0.0)
        return output


def _teacher_cache_rows(path: Path = V19_SJD_TEACHER_CACHE_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [row for row in _iter_jsonl(path) if str(row.get("status", "")).lower() == "ok"]


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=True) + "\n")


def _record_from_context_and_content(
    *,
    context: Mapping[str, Any],
    content: Mapping[str, Any],
    timestamp: pd.Timestamp,
    source: str,
    model: str,
    joiner: HistoricalJoiner,
) -> dict[str, Any]:
    feature_map = feature_map_from_context(context)
    feature_map.update(joiner.lookup(timestamp))
    market = dict(context.get("market", {}) if isinstance(context, Mapping) else {})
    simulation = dict(context.get("simulation", {}) if isinstance(context, Mapping) else {})
    sqt = dict(context.get("sqt", {}) if isinstance(context, Mapping) else {})
    current_price = _safe_float(market.get("current_price"), 0.0)
    entry_offset, sl_offset, tp_offset = _response_offsets(content, current_price=current_price)
    return {
        "timestamp": timestamp.isoformat(),
        "source": source,
        "teacher_model": model,
        "feature_map": feature_map,
        "stance": str(content.get("stance", "HOLD")).strip().upper() or "HOLD",
        "confidence": str(content.get("confidence", "LOW")).strip().upper() or "LOW",
        "entry_offset": float(entry_offset),
        "sl_offset": float(sl_offset),
        "tp_offset": float(tp_offset),
        "regime": str(simulation.get("detected_regime", "unknown")),
        "sqt_label": str(sqt.get("label", simulation.get("sqt_label", "NEUTRAL"))).strip().upper() or "NEUTRAL",
        "cabr_score": _safe_float(simulation.get("cabr_score"), 0.0),
        "mfg_disagreement": _safe_float((context.get("mfg") or {}).get("disagreement"), 0.0),
        "hurst_overall": _safe_float(simulation.get("hurst_overall"), 0.5),
    }


def _records_from_packet_logs(paths: Iterable[Path], joiner: HistoricalJoiner) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in _packet_log_paths(paths):
        for payload in _iter_jsonl(path):
            if str(payload.get("status", "")).lower() != "ok":
                continue
            if str(payload.get("request_kind", "")).strip().lower() != "kimi_judge":
                continue
            context = payload.get("context")
            content = payload.get("response_content")
            if not isinstance(context, Mapping) or not isinstance(content, Mapping):
                continue
            rows.append(
                _record_from_context_and_content(
                    context=context,
                    content=content,
                    timestamp=_timestamp_from_row(payload),
                    source=f"packet_log:{path.parent.name}",
                    model=str(payload.get("model") or payload.get("teacher_model") or ""),
                    joiner=joiner,
                )
            )
    return rows


def _records_from_teacher_cache(joiner: HistoricalJoiner) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in _teacher_cache_rows():
        context = payload.get("context")
        content = payload.get("response_content")
        if not isinstance(context, Mapping) or not isinstance(content, Mapping):
            continue
        rows.append(
            _record_from_context_and_content(
                context=context,
                content=content,
                timestamp=_timestamp_from_row(payload),
                source="teacher_cache",
                model=str(payload.get("model", "")),
                joiner=joiner,
            )
        )
    return rows


def _synthesize_teacher_rows(
    *,
    joiner: HistoricalJoiner,
    target_examples: int,
    existing_count: int,
    max_teacher_queries: int,
    symbol: str,
    teacher_models: Iterable[str],
) -> list[dict[str, Any]]:
    if existing_count >= target_examples or max_teacher_queries <= 0:
        return []
    from src.service.llm_sidecar import query_nvidia_nim

    needed = min(max(int(target_examples) - int(existing_count), 0), int(max_teacher_queries))
    sampler = SimulationContextSampler()
    contexts = sampler.sample_contexts(n_samples=max(needed, 1), balance_regimes=True, symbol=symbol)
    results: list[dict[str, Any]] = []
    for context in contexts:
        successful = None
        for model in teacher_models:
            response = query_nvidia_nim(context=context, model=model, symbol=symbol)
            if response.get("available", False) and isinstance(response.get("content"), Mapping):
                successful = response
                break
        if successful is None:
            continue
        timestamp = _timestamp_from_row({"context": context})
        cache_entry = {
            "status": "ok",
            "logged_at": timestamp.isoformat(),
            "packet_bucket_15m_utc": timestamp.floor("15min").isoformat(),
            "request_kind": "kimi_judge",
            "symbol": symbol,
            "model": successful.get("model", ""),
            "context": context,
            "response_content": successful["content"],
        }
        _append_jsonl(V19_SJD_TEACHER_CACHE_PATH, cache_entry)
        results.append(
            _record_from_context_and_content(
                context=context,
                content=successful["content"],
                timestamp=timestamp,
                source="synthetic_nim_teacher",
                model=str(successful.get("model", "")),
                joiner=joiner,
            )
        )
        if len(results) >= needed:
            break
    return results


def build_sjd_dataset(
    *,
    packet_log_paths: Iterable[Path] | None = None,
    target_examples: int = 5_000,
    max_teacher_queries: int = 0,
    symbol: str = "XAUUSD",
    teacher_models: Iterable[str] = TEACHER_MODELS,
    output_path: Path = V19_SJD_DATASET_PATH,
    feature_names_path: Path = V19_SJD_FEATURE_NAMES_PATH,
    report_path: Path = V19_SJD_DATASET_REPORT_PATH,
) -> dict[str, Any]:
    joiner = HistoricalJoiner.load()
    records = _records_from_packet_logs(packet_log_paths or [], joiner=joiner)
    records.extend(_records_from_teacher_cache(joiner=joiner))
    records.extend(
        _synthesize_teacher_rows(
            joiner=joiner,
            target_examples=int(target_examples),
            existing_count=len(records),
            max_teacher_queries=int(max_teacher_queries),
            symbol=symbol,
            teacher_models=teacher_models,
        )
    )
    if not records:
        raise RuntimeError("No SJD teacher records were available from packet logs or teacher generation.")

    feature_names = sorted({key for record in records for key in record["feature_map"].keys()})
    rows = []
    source_counts: dict[str, int] = {}
    model_counts: dict[str, int] = {}
    for record in records:
        source_counts[record["source"]] = source_counts.get(record["source"], 0) + 1
        model_key = str(record["teacher_model"] or "unknown")
        model_counts[model_key] = model_counts.get(model_key, 0) + 1
        vector = [float(record["feature_map"].get(name, 0.0)) for name in feature_names]
        rows.append(
            {
                "timestamp": record["timestamp"],
                "source": record["source"],
                "teacher_model": record["teacher_model"],
                "feature_vector": json.dumps(vector, separators=(",", ":")),
                "stance": record["stance"],
                "confidence": record["confidence"],
                "entry_offset": record["entry_offset"],
                "tp_offset": record["tp_offset"],
                "sl_offset": record["sl_offset"],
                "regime": record["regime"],
                "sqt_label": record["sqt_label"],
                "cabr_score": record["cabr_score"],
                "mfg_disagreement": record["mfg_disagreement"],
                "hurst_overall": record["hurst_overall"],
            }
        )
    frame = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_names_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    feature_names_path.write_text(json.dumps(feature_names, indent=2), encoding="utf-8")
    report = {
        "dataset_path": str(output_path),
        "feature_names_path": str(feature_names_path),
        "rows": int(len(frame)),
        "feature_count": int(len(feature_names)),
        "source_counts": source_counts,
        "teacher_model_counts": model_counts,
        "regime_counts": frame["regime"].value_counts(dropna=False).to_dict(),
        "stance_counts": frame["stance"].value_counts(dropna=False).to_dict(),
        "confidence_counts": frame["confidence"].value_counts(dropna=False).to_dict(),
        "meets_target_examples": int(len(frame)) >= int(target_examples),
        "target_examples": int(target_examples),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
