from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping

from config.project_config import (
    LLM_PROVIDER_DEFAULT,
    LM_STUDIO_BASE_URL,
    LM_STUDIO_ENABLED,
    LM_STUDIO_MODEL,
    LM_STUDIO_TIMEOUT_SECONDS,
    OLLAMA_BASE_URL,
    OLLAMA_ENABLED,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT_SECONDS,
)


@dataclass(frozen=True)
class LlmSidecarConfig:
    provider: str = LLM_PROVIDER_DEFAULT
    base_url: str = LM_STUDIO_BASE_URL
    model: str = LM_STUDIO_MODEL
    timeout_seconds: int = LM_STUDIO_TIMEOUT_SECONDS
    enabled: bool = LM_STUDIO_ENABLED


def resolve_config(provider: str | None = None, config: LlmSidecarConfig | None = None) -> LlmSidecarConfig:
    if config is not None:
        return config
    resolved_provider = (provider or LLM_PROVIDER_DEFAULT).strip().lower()
    if resolved_provider == "ollama":
        return LlmSidecarConfig(
            provider="ollama",
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL,
            timeout_seconds=OLLAMA_TIMEOUT_SECONDS,
            enabled=OLLAMA_ENABLED,
        )
    return LlmSidecarConfig(
        provider="lm_studio",
        base_url=LM_STUDIO_BASE_URL,
        model=LM_STUDIO_MODEL,
        timeout_seconds=LM_STUDIO_TIMEOUT_SECONDS,
        enabled=LM_STUDIO_ENABLED,
    )


def _post_json(url: str, payload: dict[str, Any], timeout_seconds: int) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def _get_json(url: str, timeout_seconds: int) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Content-Type": "application/json"})
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


def sidecar_health(config: LlmSidecarConfig | None = None, provider: str | None = None) -> dict[str, Any]:
    resolved = resolve_config(provider=provider, config=config)
    if not resolved.enabled:
        return {"ok": False, "enabled": False, "reason": "disabled", "provider": resolved.provider}
    try:
        if resolved.provider == "ollama":
            payload = _get_json(f"{resolved.base_url.rstrip('/')}/api/tags", timeout_seconds=resolved.timeout_seconds)
            models = payload.get("models", []) if isinstance(payload, dict) else []
            model_ids = [item.get("name") for item in models if isinstance(item, dict)]
        else:
            payload = _get_json(f"{resolved.base_url.rstrip('/')}/v1/models", timeout_seconds=resolved.timeout_seconds)
            models = payload.get("data", []) if isinstance(payload, dict) else []
            model_ids = [item.get("id") for item in models if isinstance(item, dict)]
        return {"ok": True, "enabled": True, "models": model_ids, "base_url": resolved.base_url, "active_model": resolved.model, "provider": resolved.provider}
    except Exception as exc:  # pragma: no cover
        return {"ok": False, "enabled": True, "base_url": resolved.base_url, "active_model": resolved.model, "provider": resolved.provider, "error": str(exc)}


def _chat_json_request(system_prompt: str, user_prompt: str, config: LlmSidecarConfig | None = None, provider: str | None = None) -> dict[str, Any]:
    resolved = resolve_config(provider=provider, config=config)
    if not resolved.enabled:
        return {"available": False, "reason": "disabled", "provider": resolved.provider}

    try:
        if resolved.provider == "ollama":
            payload = {
                "model": resolved.model,
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
                "model": resolved.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
            }
            response = _post_json(f"{resolved.base_url.rstrip('/')}/v1/chat/completions", payload, timeout_seconds=resolved.timeout_seconds)
            choices = response.get("choices", []) if isinstance(response, dict) else []
            if not choices:
                raise ValueError("No chat choices returned by LM Studio")
            message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
            content = message.get("content", "")
        parsed = parse_json_text(content)
        return {"available": True, "model": resolved.model, "base_url": resolved.base_url, "provider": resolved.provider, "content": parsed}
    except Exception as exc:  # pragma: no cover
        return {"available": False, "model": resolved.model, "base_url": resolved.base_url, "provider": resolved.provider, "error": str(exc)}


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


def request_market_context(symbol: str, context: Mapping[str, Any], config: LlmSidecarConfig | None = None, provider: str | None = None) -> dict[str, Any]:
    prompt = build_market_context_prompt(symbol, context)
    return _chat_json_request(
        system_prompt="You are a strict JSON market-context extractor.",
        user_prompt=prompt,
        config=config,
        provider=provider,
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


def request_swarm_judgment(symbol: str, context: Mapping[str, Any], config: LlmSidecarConfig | None = None, provider: str | None = None) -> dict[str, Any]:
    prompt = build_swarm_judgment_prompt(symbol, context)
    return _chat_json_request(
        system_prompt="You are a strict JSON swarm-judgment engine.",
        user_prompt=prompt,
        config=config,
        provider=provider,
    )
