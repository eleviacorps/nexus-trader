from __future__ import annotations

from collections import deque
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
import urllib.error
import urllib.request
from dataclasses import dataclass
import re
from typing import Any, Callable, Mapping

from config.project_config import (
    LLM_PROVIDER_DEFAULT,
    LM_STUDIO_BASE_URL,
    LM_STUDIO_ENABLED,
    LM_STUDIO_MODEL,
    LM_STUDIO_TIMEOUT_SECONDS,
    NVIDIA_NIM_API_KEY,
    NVIDIA_NIM_BASE_URL,
    NVIDIA_NIM_ENABLED,
    NVIDIA_NIM_MAX_REQUESTS_PER_MINUTE,
    NVIDIA_NIM_MODEL,
    NVIDIA_NIM_TIMEOUT_SECONDS,
    V17_KIMI_PACKET_LOG_PATH,
    V18_KIMI_PACKET_LOG_PATH,
    V19_SJD_MODEL_NPZ_PATH,
    V19_SJD_MODEL_PATH,
    OLLAMA_BASE_URL,
    OLLAMA_ENABLED,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT_SECONDS,
)
from src.v18.kimi_system_prompt import KIMI_SYSTEM_PROMPT, build_kimi_user_message


@dataclass(frozen=True)
class LlmSidecarConfig:
    provider: str = LLM_PROVIDER_DEFAULT
    base_url: str = LM_STUDIO_BASE_URL
    model: str = LM_STUDIO_MODEL
    timeout_seconds: int = LM_STUDIO_TIMEOUT_SECONDS
    enabled: bool = LM_STUDIO_ENABLED
    api_key: str = ""


NVIDIA_NIM_MODEL_FALLBACK_CHAIN = [
    "moonshotai/kimi-k2-instruct",
    "moonshotai/kimi-k2-5",
    "nvidia/llama-3.3-nemotron-super-49b-v1",
    "meta/llama-3.1-70b-instruct",
]


NUMERIC_FIELD_MEANINGS: dict[str, str] = {
    "market.current_price": "Latest observed market price for the instrument at request time.",
    "market.atr_14": "14-bar average true range used as the local volatility scale.",
    "simulation.overall_confidence": "Composite 15-minute signal strength from the live simulator, on a 0 to 1 scale.",
    "simulation.cabr_score": "Cross-attention branch-ranker confidence proxy; higher means the selected path is stronger.",
    "simulation.cpm_score": "Conditional predictability score; higher means the local bar state has been more repeatable historically.",
    "simulation.cone_width_pips": "Width of the inner 15-minute prediction cone in pips; larger means more uncertainty.",
    "simulation.cone_c_m": "Empirical relativistic cone speed limit used to define the outer hard price envelope.",
    "simulation.testosterone_index.retail": "WLTC winner-cycle intensity for the retail crowd.",
    "simulation.testosterone_index.institutional": "WLTC winner-cycle intensity for the institutional crowd.",
    "simulation.macro_bias": "Macro directional tilt normalized to roughly -1 bearish through +1 bullish.",
    "simulation.news_bias": "News sentiment tilt normalized to roughly -1 bearish through +1 bullish.",
    "simulation.crowd_bias": "Crowd sentiment tilt normalized to roughly -1 bearish through +1 bullish.",
    "simulation.crowd_extreme": "How emotionally stretched the crowd proxy is, from 0 calm to 1 extreme.",
    "simulation.hurst_overall": "Overall Hurst exponent; above 0.5 implies persistence, below 0.5 implies mean reversion.",
    "simulation.hurst_positive": "Positive-move Hurst exponent describing upward persistence.",
    "simulation.hurst_negative": "Negative-move Hurst exponent describing downward persistence.",
    "simulation.hurst_asymmetry": "Difference between upward and downward persistence, H+ minus H-.",
    "simulation.eci_cone_width_modifier": "Economic-calendar cone widening factor applied around high-impact events.",
    "technical_analysis.quant_regime_strength": "Quant stack estimate of how coherent the current regime is.",
    "technical_analysis.quant_transition_risk": "Risk that the current regime is about to change.",
    "technical_analysis.quant_vol_realism": "How realistic current volatility looks relative to historical analog structure.",
    "technical_analysis.quant_fair_value_z": "Distance from estimated fair value in standard-deviation units.",
    "technical_analysis.rsi_14": "Fourteen-bar RSI momentum oscillator.",
    "technical_analysis.atr_14": "Fourteen-bar average true range in raw price units.",
    "technical_analysis.equilibrium": "Midpoint of the recent dealing range used to classify premium versus discount.",
    "bot_swarm.aggregate.bullish_probability": "Aggregate bullish probability from the specialist bot swarm.",
    "bot_swarm.aggregate.disagreement": "How much the specialist bots disagree with one another.",
    "sqt.rolling_accuracy": "Rolling recent hit rate of the simulator.",
    "current_row.macro_bias": "Macro directional tilt for the live bar context.",
    "current_row.macro_shock": "Macro shock intensity for the live bar context.",
    "current_row.news_bias": "News directional tilt for the live bar context.",
    "current_row.news_intensity": "News-event intensity score for the live bar context.",
    "current_row.crowd_bias": "Crowd directional tilt for the live bar context.",
    "current_row.crowd_extreme": "Crowd emotional extremeness from 0 to 1.",
    "current_row.analog_bias": "Historical analog directional tilt from -1 bearish to +1 bullish.",
    "current_row.analog_confidence": "Confidence in the analog retrieval match from 0 to 1.",
    "current_row.close": "Latest close price used as the simulation anchor.",
    "current_row.atr_14": "14-bar ATR at the simulation anchor.",
    "current_row.hurst_overall": "Live MMM overall Hurst feature.",
    "current_row.hurst_positive": "Live MMM positive-side Hurst feature.",
    "current_row.hurst_negative": "Live MMM negative-side Hurst feature.",
    "current_row.hurst_asymmetry": "Live MMM asymmetry feature, H+ minus H-.",
    "current_row.wltc_testosterone_retail": "Retail persona testosterone index from WLTC; higher means stronger winner-cycle behavior.",
    "current_row.wltc_testosterone_noise": "Noise persona testosterone index from WLTC; higher means more impulsive chase behavior.",
    "current_row.wltc_fundamental_tracking_retail": "Retail fair-value tracking weight after WLTC adjustment.",
    "current_row.wltc_fundamental_tracking_institutional": "Institutional fair-value tracking weight after WLTC adjustment.",
    "eci.cone_width_modifier": "Event-risk modifier that widens the V17 cone near major releases.",
    "eci.mins_to_next_high": "Minutes until the next high-impact event.",
    "eci.mins_since_last_high": "Minutes since the last high-impact event.",
    "mfg.disagreement": "Cross-persona disagreement in expected drift; higher values imply more conflict between sub-populations.",
    "mfg.consensus_drift": "Weighted mean-field drift estimate implied by the persona belief states.",
}

_LOCAL_SJD_BUNDLE: Any | None = None
_LOCAL_SJD_LOAD_ERROR: str | None = None
_NIM_REQUEST_TIMESTAMPS: deque[float] = deque()
_NIM_RATE_LIMIT_LOCK = RLock()


def is_nvidia_nim_provider(provider: str | None) -> bool:
    return (provider or "").strip().lower() in {"nvidia_nim", "nim", "kimi", "qwen"}


def get_nim_model(requested_model: str | None) -> str:
    candidate = (requested_model or "").strip()
    if candidate:
        return candidate
    return NVIDIA_NIM_MODEL_FALLBACK_CHAIN[0]


def _nim_model_chain(requested_model: str | None) -> list[str]:
    explicit = (requested_model or "").strip()
    if explicit:
        return [explicit]
    chain = [get_nim_model(requested_model)]
    chain.extend(NVIDIA_NIM_MODEL_FALLBACK_CHAIN)
    output: list[str] = []
    seen: set[str] = set()
    for item in chain:
        key = item.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(key)
    return output


def resolve_config(
    provider: str | None = None,
    config: LlmSidecarConfig | None = None,
    model: str | None = None,
) -> LlmSidecarConfig:
    if config is not None:
        return config
    resolved_provider = (provider or LLM_PROVIDER_DEFAULT).strip().lower()
    if resolved_provider == "ollama":
        return LlmSidecarConfig(
            provider="ollama",
            base_url=OLLAMA_BASE_URL,
            model=model or OLLAMA_MODEL,
            timeout_seconds=OLLAMA_TIMEOUT_SECONDS,
            enabled=OLLAMA_ENABLED,
        )
    if resolved_provider in {"nvidia_nim", "nim", "kimi", "qwen"}:
        return LlmSidecarConfig(
            provider="nvidia_nim",
            base_url=NVIDIA_NIM_BASE_URL,
            model=get_nim_model(model or NVIDIA_NIM_MODEL),
            timeout_seconds=NVIDIA_NIM_TIMEOUT_SECONDS,
            enabled=NVIDIA_NIM_ENABLED,
            api_key=NVIDIA_NIM_API_KEY,
        )
    return LlmSidecarConfig(
        provider="lm_studio",
        base_url=LM_STUDIO_BASE_URL,
        model=model or LM_STUDIO_MODEL,
        timeout_seconds=LM_STUDIO_TIMEOUT_SECONDS,
        enabled=LM_STUDIO_ENABLED,
    )


def _request_headers(api_key: str = "") -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _openai_compat_url(base_url: str, endpoint: str) -> str:
    base = (base_url or "").rstrip("/")
    suffix = endpoint if endpoint.startswith("/") else f"/{endpoint}"
    if base.endswith("/v1"):
        return f"{base}{suffix}"
    return f"{base}/v1{suffix}"


def _flatten_numeric_paths(payload: Any, prefix: str = "") -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_numeric_paths(value, child))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            rows.extend(_flatten_numeric_paths(value, f"{prefix}[{index}]"))
    else:
        if isinstance(payload, bool):
            return rows
        if isinstance(payload, (int, float)):
            rows.append((prefix, float(payload)))
    return rows


def build_numeric_glossary(context: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    glossary: dict[str, dict[str, Any]] = {}
    for path, value in _flatten_numeric_paths(context):
        glossary[path] = {
            "value": round(float(value), 6),
            "meaning": NUMERIC_FIELD_MEANINGS.get(path, f"Numeric V17 context value carried under `{path}`."),
        }
    return glossary


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        if number != number or number in {float("inf"), float("-inf")}:
            return float(default)
        return number
    except Exception:
        return float(default)


def _packet_bucket_utc(timestamp: datetime | None = None) -> str:
    now = timestamp or datetime.now(timezone.utc)
    bucket_minute = (now.minute // 15) * 15
    snapped = now.replace(minute=bucket_minute, second=0, microsecond=0)
    return snapped.isoformat()


def _nim_rate_limit_snapshot(*, now_monotonic: float | None = None) -> dict[str, Any]:
    limit = max(int(NVIDIA_NIM_MAX_REQUESTS_PER_MINUTE), 1)
    now_value = float(now_monotonic if now_monotonic is not None else time.monotonic())
    with _NIM_RATE_LIMIT_LOCK:
        while _NIM_REQUEST_TIMESTAMPS and (now_value - _NIM_REQUEST_TIMESTAMPS[0]) >= 60.0:
            _NIM_REQUEST_TIMESTAMPS.popleft()
        used = len(_NIM_REQUEST_TIMESTAMPS)
        remaining = max(limit - used, 0)
        wait_seconds = 0.0
        if used >= limit and _NIM_REQUEST_TIMESTAMPS:
            wait_seconds = max(0.0, 60.0 - (now_value - _NIM_REQUEST_TIMESTAMPS[0]))
    return {
        "limit_per_minute": limit,
        "used_in_current_window": used,
        "remaining_in_current_window": remaining,
        "retry_after_seconds": round(float(wait_seconds), 3),
    }


def nim_rate_limit_snapshot() -> dict[str, Any]:
    return _nim_rate_limit_snapshot()


def _reserve_nim_request_slot() -> tuple[bool, dict[str, Any]]:
    limit = max(int(NVIDIA_NIM_MAX_REQUESTS_PER_MINUTE), 1)
    now_value = float(time.monotonic())
    with _NIM_RATE_LIMIT_LOCK:
        while _NIM_REQUEST_TIMESTAMPS and (now_value - _NIM_REQUEST_TIMESTAMPS[0]) >= 60.0:
            _NIM_REQUEST_TIMESTAMPS.popleft()
        if len(_NIM_REQUEST_TIMESTAMPS) >= limit:
            snapshot = _nim_rate_limit_snapshot(now_monotonic=now_value)
            return False, snapshot
        _NIM_REQUEST_TIMESTAMPS.append(now_value)
        snapshot = _nim_rate_limit_snapshot(now_monotonic=now_value)
    return True, snapshot


def _append_packet_log(entry: Mapping[str, Any], path: Path = V17_KIMI_PACKET_LOG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(entry), ensure_ascii=True) + "\n")


def read_packet_log(limit: int = 20, path: Path = V17_KIMI_PACKET_LOG_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows[-max(int(limit), 1) :]


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _load_local_sjd_bundle() -> Any | None:
    global _LOCAL_SJD_BUNDLE, _LOCAL_SJD_LOAD_ERROR
    if _LOCAL_SJD_BUNDLE is not None:
        return _LOCAL_SJD_BUNDLE
    if _LOCAL_SJD_LOAD_ERROR is not None:
        return None
    if not V19_SJD_MODEL_PATH.exists() and not V19_SJD_MODEL_NPZ_PATH.exists():
        _LOCAL_SJD_LOAD_ERROR = "missing_v19_sjd_checkpoint"
        return None
    try:
        if V19_SJD_MODEL_NPZ_PATH.exists():
            from src.v19.sjd_numpy import load_sjd_npz_bundle

            _LOCAL_SJD_BUNDLE = load_sjd_npz_bundle(path=V19_SJD_MODEL_NPZ_PATH)
        else:
            from src.v19.sjd_model import load_sjd_bundle

            _LOCAL_SJD_BUNDLE = load_sjd_bundle(path=V19_SJD_MODEL_PATH, device="cpu")
        return _LOCAL_SJD_BUNDLE
    except Exception as exc:
        _LOCAL_SJD_LOAD_ERROR = str(exc)
        return None


def _predict_local_sjd(symbol: str, context: Mapping[str, Any]) -> dict[str, Any] | None:
    bundle = _load_local_sjd_bundle()
    if bundle is None:
        return None
    try:
        if hasattr(bundle, "weights"):
            from src.v19.sjd_numpy import predict_sjd_from_context_numpy

            content = predict_sjd_from_context_numpy(bundle, context, symbol=symbol, pip_size=0.1)
        else:
            from src.v19.sjd_model import predict_sjd_from_context

            content = predict_sjd_from_context(bundle, context, symbol=symbol, pip_size=0.1)
        return {
            "available": True,
            "provider": "local_sjd",
            "model": "v19_sjd_local",
            "content": content,
            "source": "local_sjd",
        }
    except Exception:
        return None


def request_local_sjd_judge(
    symbol: str,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    local_sjd = _predict_local_sjd(symbol, context)
    if local_sjd is not None:
        return local_sjd
    return {
        "available": False,
        "provider": "local_sjd",
        "model": "v19_sjd_local",
        "error": _LOCAL_SJD_LOAD_ERROR or "local_sjd_unavailable",
    }


def _cached_row_similarity(context: Mapping[str, Any], row: Mapping[str, Any]) -> float:
    candidate_context = row.get("context", {}) if isinstance(row, Mapping) else {}
    if not isinstance(candidate_context, Mapping):
        return float("inf")
    left_market = dict(context.get("market", {}) if isinstance(context, Mapping) else {})
    right_market = dict(candidate_context.get("market", {}) if isinstance(candidate_context, Mapping) else {})
    left_sim = dict(context.get("simulation", {}) if isinstance(context, Mapping) else {})
    right_sim = dict(candidate_context.get("simulation", {}) if isinstance(candidate_context, Mapping) else {})
    left_mfg = dict(context.get("mfg", {}) if isinstance(context, Mapping) else {})
    right_mfg = dict(candidate_context.get("mfg", {}) if isinstance(candidate_context, Mapping) else {})
    price_gap = abs(_safe_float(left_market.get("current_price"), 0.0) - _safe_float(right_market.get("current_price"), 0.0))
    cabr_gap = abs(_safe_float(left_sim.get("cabr_score"), 0.0) - _safe_float(right_sim.get("cabr_score"), 0.0))
    cpm_gap = abs(_safe_float(left_sim.get("cpm_score"), 0.0) - _safe_float(right_sim.get("cpm_score"), 0.0))
    hurst_gap = abs(_safe_float(left_sim.get("hurst_asymmetry"), 0.0) - _safe_float(right_sim.get("hurst_asymmetry"), 0.0))
    disagreement_gap = abs(_safe_float(left_mfg.get("disagreement"), 0.0) - _safe_float(right_mfg.get("disagreement"), 0.0))
    left_direction = str(left_sim.get("direction", left_sim.get("scenario_bias", "HOLD"))).upper()
    right_direction = str(right_sim.get("direction", right_sim.get("scenario_bias", "HOLD"))).upper()
    direction_penalty = 0.0 if left_direction == right_direction else 0.5
    left_regime = str(left_sim.get("detected_regime", "unknown")).lower()
    right_regime = str(right_sim.get("detected_regime", "unknown")).lower()
    regime_penalty = 0.0 if left_regime == right_regime else 0.25
    return price_gap + (10.0 * cabr_gap) + (10.0 * cpm_gap) + (5.0 * hurst_gap) + (1000.0 * disagreement_gap) + direction_penalty + regime_penalty


def _cached_packet_fallback(symbol: str, context: Mapping[str, Any]) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for path in (V18_KIMI_PACKET_LOG_PATH, V17_KIMI_PACKET_LOG_PATH):
        for row in _read_jsonl_rows(path):
            if str(row.get("status", "")).lower() != "ok":
                continue
            content = row.get("response_content")
            if not isinstance(content, Mapping):
                continue
            if row.get("request_kind") not in {None, "", "kimi_judge"}:
                continue
            candidates.append({"path": path, "row": row})
    if not candidates:
        return None
    best = min(candidates, key=lambda item: _cached_row_similarity(context, item["row"]))
    content = _normalize_kimi_fields(best["row"].get("response_content", {}))
    return {
        "available": True,
        "provider": "cached_packet",
        "model": str(best["row"].get("model", "")),
        "content": content,
        "source": f"cached_packet:{best['path'].parent.name}",
    }


def _post_json(url: str, payload: dict[str, Any], timeout_seconds: int, api_key: str = "") -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers=_request_headers(api_key=api_key),
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def _get_json(url: str, timeout_seconds: int, api_key: str = "") -> dict[str, Any]:
    request = urllib.request.Request(url, headers=_request_headers(api_key=api_key))
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_json_text(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if not text:
        raise ValueError("Empty LLM response")
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("Expected JSON object from LLM")
    return payload


def _normalize_kimi_summary(block: Any, fallback_call: str, fallback_summary: str, fallback_reasoning: str) -> dict[str, str]:
    if isinstance(block, Mapping):
        call = str(block.get("call", fallback_call)).strip().upper() or fallback_call
        if call == "HOLD":
            call = "SKIP"
        return {
            "call": call,
            "summary": str(block.get("summary", fallback_summary)).strip() or fallback_summary,
            "reasoning": str(block.get("reasoning", fallback_reasoning)).strip() or fallback_reasoning,
        }
    return {"call": fallback_call, "summary": fallback_summary, "reasoning": fallback_reasoning}


def _normalize_kimi_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(payload)
    stance = str(output.get("stance", "HOLD")).strip().upper() or "HOLD"
    if stance not in {"BUY", "SELL", "HOLD"}:
        stance = "HOLD"
    final_call = str(output.get("final_call", stance)).strip().upper() or stance
    if final_call == "HOLD":
        final_call = "SKIP"
    if final_call not in {"BUY", "SELL", "SKIP"}:
        final_call = "SKIP"
    reasoning = str(output.get("reasoning", "No Kimi reasoning supplied.")).strip() or "No Kimi reasoning supplied."
    output["stance"] = stance
    output["confidence"] = str(output.get("confidence", "VERY_LOW")).strip().upper() or "VERY_LOW"
    output["final_call"] = final_call
    output["final_summary"] = str(output.get("final_summary", f"{final_call} - {reasoning}")).strip() or f"{final_call} - {reasoning}"
    output["market_only_summary"] = _normalize_kimi_summary(
        output.get("market_only_summary"),
        final_call,
        f"Market-only read is {final_call}.",
        "This block was not returned explicitly by the model, so the main reasoning is being reused.",
    )
    output["v18_summary"] = _normalize_kimi_summary(
        output.get("v18_summary"),
        final_call,
        f"V18-only read is {final_call}.",
        "This block was not returned explicitly by the model, so the main reasoning is being reused.",
    )
    output["combined_summary"] = _normalize_kimi_summary(
        output.get("combined_summary"),
        final_call,
        f"Combined read is {final_call}.",
        reasoning,
    )
    output["reasoning"] = reasoning
    output["key_risk"] = str(output.get("key_risk", "No risk note supplied.")).strip() or "No risk note supplied."
    output["crowd_note"] = str(output.get("crowd_note", "No crowd note supplied.")).strip() or "No crowd note supplied."
    output["regime_note"] = str(output.get("regime_note", "No regime note supplied.")).strip() or "No regime note supplied."
    return output


def parse_kimi_response(raw_response: str) -> dict[str, Any]:
    clean = re.sub(r"```(?:json)?", "", str(raw_response or ""), flags=re.IGNORECASE).strip().strip("`").strip()
    if clean:
        try:
            payload = parse_json_text(clean)
            return _normalize_kimi_fields(payload)
        except Exception:
            pass
    return _normalize_kimi_fields(
        {
        "stance": "HOLD",
        "confidence": "VERY_LOW",
        "entry_zone": [],
        "stop_loss": None,
        "take_profit": None,
        "hold_time": "skip",
        "reasoning": f"Response parsing failed: {clean[:200]}",
        "key_risk": "Unable to parse model response",
        "crowd_note": "Unavailable due to parsing failure.",
        "regime_note": "Unavailable due to parsing failure.",
        "invalidation": None,
        "error": True,
        }
    )


def sidecar_health(
    config: LlmSidecarConfig | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    resolved = resolve_config(provider=provider, config=config, model=model)
    if not resolved.enabled:
        return {"ok": False, "enabled": False, "reason": "disabled", "provider": resolved.provider}
    try:
        if resolved.provider == "ollama":
            payload = _get_json(f"{resolved.base_url.rstrip('/')}/api/tags", timeout_seconds=resolved.timeout_seconds)
            models = payload.get("models", []) if isinstance(payload, dict) else []
            model_ids = [item.get("name") for item in models if isinstance(item, dict)]
        else:
            payload = _get_json(
                _openai_compat_url(resolved.base_url, "/models"),
                timeout_seconds=resolved.timeout_seconds,
                api_key=resolved.api_key,
            )
            models = payload.get("data", []) if isinstance(payload, dict) else []
            model_ids = [item.get("id") for item in models if isinstance(item, dict)]
        payload = {"ok": True, "enabled": True, "models": model_ids, "base_url": resolved.base_url, "active_model": resolved.model, "provider": resolved.provider}
        if resolved.provider == "nvidia_nim":
            payload["rate_limit"] = nim_rate_limit_snapshot()
        return payload
    except Exception as exc:  # pragma: no cover
        payload = {"ok": False, "enabled": True, "base_url": resolved.base_url, "active_model": resolved.model, "provider": resolved.provider, "error": str(exc)}
        if resolved.provider == "nvidia_nim":
            payload["rate_limit"] = nim_rate_limit_snapshot()
        return payload


def _chat_json_request(
    system_prompt: str,
    user_prompt: str,
    config: LlmSidecarConfig | None = None,
    provider: str | None = None,
    model: str | None = None,
    request_kind: str = "generic",
    symbol: str | None = None,
    context: Mapping[str, Any] | None = None,
    response_parser: Callable[[str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    resolved = resolve_config(provider=provider, config=config, model=model)
    if not resolved.enabled:
        return {"available": False, "reason": "disabled", "provider": resolved.provider}

    should_log = resolved.provider == "nvidia_nim"
    packet_context = dict(context or {})
    attempted_models = _nim_model_chain(model or resolved.model) if resolved.provider == "nvidia_nim" else [resolved.model]
    packet_entry = {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "packet_bucket_15m_utc": _packet_bucket_utc(),
        "request_kind": str(request_kind),
        "symbol": str(symbol or packet_context.get("symbol", "")),
        "provider": resolved.provider,
        "model": resolved.model,
        "attempted_models": attempted_models,
        "base_url": resolved.base_url,
        "horizon_minutes": 15,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "context": packet_context,
        "numeric_glossary": build_numeric_glossary(packet_context),
    }
    errors: list[str] = []
    for chosen_model in attempted_models:
        try:
            if resolved.provider == "ollama":
                payload = {
                    "model": chosen_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                }
                response = _post_json(f"{resolved.base_url.rstrip('/')}/api/chat", payload, timeout_seconds=resolved.timeout_seconds)
                message = response.get("message", {}) if isinstance(response, dict) else {}
                content = message.get("content", "")
            else:
                rate_limit_snapshot = None
                if resolved.provider == "nvidia_nim":
                    allowed, rate_limit_snapshot = _reserve_nim_request_slot()
                    if not allowed:
                        raise RuntimeError(
                            "local_nim_rate_limit_guard:"
                            f" limit={rate_limit_snapshot['limit_per_minute']}/min"
                            f" used={rate_limit_snapshot['used_in_current_window']}"
                            f" retry_after={rate_limit_snapshot['retry_after_seconds']}s"
                        )
                payload = {
                    "model": chosen_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "max_tokens": 4096,
                    "stream": False,
                }
                response = _post_json(
                    _openai_compat_url(resolved.base_url, "/chat/completions"),
                    payload,
                    timeout_seconds=resolved.timeout_seconds,
                    api_key=resolved.api_key,
                )
                choices = response.get("choices", []) if isinstance(response, dict) else []
                if not choices:
                    raise ValueError("No chat choices returned by provider.")
                message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
                content = message.get("content", "")
            parser = response_parser or parse_json_text
            parsed = parser(content)
            if should_log:
                _append_packet_log(packet_entry | {"status": "ok", "model": chosen_model, "raw_response": content, "response_content": parsed})
            return {
                "available": True,
                "model": chosen_model,
                "base_url": resolved.base_url,
                "provider": resolved.provider,
                "content": parsed,
                "rate_limit": nim_rate_limit_snapshot() if resolved.provider == "nvidia_nim" else None,
            }
        except Exception as exc:  # pragma: no cover
            errors.append(f"{chosen_model}: {exc}")
            if resolved.provider != "nvidia_nim":
                break
    error_text = " | ".join(errors) if errors else "Unknown sidecar error"
    if should_log:
        _append_packet_log(packet_entry | {"status": "error", "error": error_text})
    return {
        "available": False,
        "model": attempted_models[0] if attempted_models else resolved.model,
        "base_url": resolved.base_url,
        "provider": resolved.provider,
        "error": error_text,
        "rate_limit": nim_rate_limit_snapshot() if resolved.provider == "nvidia_nim" else None,
    }


def build_market_context_prompt(symbol: str, context: Mapping[str, Any]) -> str:
    return f"""
You are a market-interpretation sidecar for Nexus Trader.
Return only valid JSON with no markdown and no prose.

Task:
- Read the supplied market state for {symbol}
- Summarize narrative and regime context
- Do not provide direct trading advice
- Do not provide a direct numeric price target
- Keep all scores between -1.0 and 1.0 unless otherwise specified

Required JSON schema:
{{
  "macro_thesis": "short string",
  "dominant_narrative": "short string",
  "event_severity": 0.0,
  "risk_of_regime_shift": 0.0,
  "institutional_bias": 0.0,
  "whale_bias": 0.0,
  "retail_bias": 0.0,
  "confidence_note": "short string",
  "explanation": "1-3 short sentences"
}}

Current market state:
{json.dumps(context, ensure_ascii=True)}
""".strip()


def request_market_context(
    symbol: str,
    context: Mapping[str, Any],
    config: LlmSidecarConfig | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    prompt = build_market_context_prompt(symbol, context)
    return _chat_json_request(
        system_prompt="You are a strict JSON market-context extractor.",
        user_prompt=prompt,
        config=config,
        provider=provider,
        model=model,
        request_kind="market_context",
        symbol=symbol,
        context=context,
    )


def build_swarm_judgment_prompt(symbol: str, context: Mapping[str, Any]) -> str:
    return f"""
You are the Nexus Trader swarm judge.
Return only valid JSON with no markdown and no prose outside JSON.

Task:
- Read the specialist bot outputs, simulator result, branch summaries, and technical context for {symbol}
- Judge the disagreement between them
- Summarize crowd lean, likely emotion, and what the minority scenario is
- Provide a final manual stance for a human observer
- Do not invent a numeric price target

Required JSON schema:
{{
  "master_bias": "bullish|bearish|neutral",
  "master_confidence": 0.0,
  "manual_stance": "buy|sell|hold",
  "manual_action_reason": "short string",
  "crowd_emotion": "short string",
  "crowd_lean": "short string",
  "discussion_takeaway": "short string",
  "top_bot": "short string",
  "weakest_bot": "short string",
  "judge_summary": "1-3 short sentences",
  "debate_lines": ["line1", "line2", "line3", "line4"],
  "public_reaction_lines": ["line1", "line2", "line3"],
  "minority_case": "short string",
  "actionable_structure": "short string"
}}

Current swarm state:
{json.dumps(context, ensure_ascii=True)}
""".strip()


def request_swarm_judgment(
    symbol: str,
    context: Mapping[str, Any],
    config: LlmSidecarConfig | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    prompt = build_swarm_judgment_prompt(symbol, context)
    return _chat_json_request(
        system_prompt="You are a strict JSON swarm-judgment engine.",
        user_prompt=prompt,
        config=config,
        provider=provider,
        model=model,
        request_kind="swarm_judgment",
        symbol=symbol,
        context=context,
    )


def query_nvidia_nim(
    *,
    context: Mapping[str, Any],
    model: str | None = None,
    symbol: str = "XAUUSD",
    config: LlmSidecarConfig | None = None,
) -> dict[str, Any]:
    return _chat_json_request(
        system_prompt=KIMI_SYSTEM_PROMPT,
        user_prompt=build_kimi_user_message(context, symbol),
        config=config,
        provider="nvidia_nim",
        model=model,
        request_kind="kimi_judge",
        symbol=symbol,
        context=context,
        response_parser=parse_kimi_response,
    )


def request_kimi_judge(
    symbol: str,
    context: Mapping[str, Any],
    config: LlmSidecarConfig | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    nim_response = query_nvidia_nim(
        context=context,
        model=model,
        symbol=symbol,
        config=config if is_nvidia_nim_provider(provider or "nvidia_nim") else None,
    )
    if nim_response.get("available", False):
        return nim_response | {"source": "nvidia_nim"}
    cached = _cached_packet_fallback(symbol, context)
    if cached is not None:
        return cached
    return nim_response
