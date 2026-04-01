from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping

from config.project_config import LM_STUDIO_BASE_URL, LM_STUDIO_ENABLED, LM_STUDIO_MODEL, LM_STUDIO_TIMEOUT_SECONDS


@dataclass(frozen=True)
class LlmSidecarConfig:
    base_url: str = LM_STUDIO_BASE_URL
    model: str = LM_STUDIO_MODEL
    timeout_seconds: int = LM_STUDIO_TIMEOUT_SECONDS
    enabled: bool = LM_STUDIO_ENABLED


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


def sidecar_health(config: LlmSidecarConfig | None = None) -> dict[str, Any]:
    resolved = config or LlmSidecarConfig()
    if not resolved.enabled:
        return {"ok": False, "enabled": False, "reason": "disabled"}
    try:
        payload = _get_json(f"{resolved.base_url.rstrip('/')}/v1/models", timeout_seconds=resolved.timeout_seconds)
        models = payload.get("data", []) if isinstance(payload, dict) else []
        model_ids = [item.get("id") for item in models if isinstance(item, dict)]
        return {"ok": True, "enabled": True, "models": model_ids, "base_url": resolved.base_url, "active_model": resolved.model}
    except Exception as exc:  # pragma: no cover
        return {"ok": False, "enabled": True, "base_url": resolved.base_url, "active_model": resolved.model, "error": str(exc)}


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


def request_market_context(symbol: str, context: Mapping[str, Any], config: LlmSidecarConfig | None = None) -> dict[str, Any]:
    resolved = config or LlmSidecarConfig()
    if not resolved.enabled:
        return {"available": False, "reason": "disabled"}

    prompt = build_market_context_prompt(symbol, context)
    payload = {
        "model": resolved.model,
        "messages": [
            {"role": "system", "content": "You are a strict JSON market-context extractor."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    try:
        response = _post_json(f"{resolved.base_url.rstrip('/')}/v1/chat/completions", payload, timeout_seconds=resolved.timeout_seconds)
        choices = response.get("choices", []) if isinstance(response, dict) else []
        if not choices:
            raise ValueError("No chat choices returned by LM Studio")
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content = message.get("content", "")
        parsed = parse_json_text(content)
        return {"available": True, "model": resolved.model, "base_url": resolved.base_url, "content": parsed}
    except Exception as exc:  # pragma: no cover
        return {"available": False, "model": resolved.model, "base_url": resolved.base_url, "error": str(exc)}
