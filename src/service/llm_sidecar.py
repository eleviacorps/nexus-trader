from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
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
    NVIDIA_NIM_MODEL,
    NVIDIA_NIM_TIMEOUT_SECONDS,
    V17_KIMI_PACKET_LOG_PATH,
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
    "nvidia/llama-3.1-nemotron-70b-instruct",
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


def is_nvidia_nim_provider(provider: str | None) -> bool:
    return (provider or "").strip().lower() in {"nvidia_nim", "nim", "kimi", "qwen"}


def get_nim_model(requested_model: str | None) -> str:
    candidate = (requested_model or "").strip()
    if candidate:
        return candidate
    return NVIDIA_NIM_MODEL_FALLBACK_CHAIN[0]


def _nim_model_chain(requested_model: str | None) -> list[str]:
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


def _packet_bucket_utc(timestamp: datetime | None = None) -> str:
    now = timestamp or datetime.now(timezone.utc)
    bucket_minute = (now.minute // 15) * 15
    snapped = now.replace(minute=bucket_minute, second=0, microsecond=0)
    return snapped.isoformat()


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


def parse_kimi_response(raw_response: str) -> dict[str, Any]:
    clean = re.sub(r"```(?:json)?", "", str(raw_response or ""), flags=re.IGNORECASE).strip().strip("`").strip()
    if clean:
        try:
            payload = parse_json_text(clean)
            return payload
        except Exception:
            pass
    return {
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
        return {"ok": True, "enabled": True, "models": model_ids, "base_url": resolved.base_url, "active_model": resolved.model, "provider": resolved.provider}
    except Exception as exc:  # pragma: no cover
        return {"ok": False, "enabled": True, "base_url": resolved.base_url, "active_model": resolved.model, "provider": resolved.provider, "error": str(exc)}


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


def request_kimi_judge(
    symbol: str,
    context: Mapping[str, Any],
    config: LlmSidecarConfig | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    return _chat_json_request(
        system_prompt=KIMI_SYSTEM_PROMPT,
        user_prompt=build_kimi_user_message(context, symbol),
        config=config,
        provider=provider or "nvidia_nim",
        model=model,
        request_kind="kimi_judge",
        symbol=symbol,
        context=context,
        response_parser=parse_kimi_response,
    )
