from __future__ import annotations

import unittest

from src.service.llm_sidecar import _nim_model_chain, _openai_compat_url, nim_rate_limit_snapshot, parse_json_text, resolve_config


class LlmSidecarTests(unittest.TestCase):
    def test_parse_json_text_accepts_plain_json(self):
        payload = parse_json_text('{"macro_thesis":"steady","event_severity":0.2}')
        self.assertEqual(payload["macro_thesis"], "steady")
        self.assertEqual(payload["event_severity"], 0.2)

    def test_parse_json_text_accepts_fenced_json(self):
        payload = parse_json_text("```json\n{\"dominant_narrative\":\"dollar firming\"}\n```")
        self.assertEqual(payload["dominant_narrative"], "dollar firming")

    def test_resolve_config_supports_ollama_provider(self):
        config = resolve_config(provider="ollama")
        self.assertEqual(config.provider, "ollama")

    def test_openai_compat_url_avoids_duplicate_v1(self):
        self.assertEqual(
            _openai_compat_url("https://integrate.api.nvidia.com/v1", "/chat/completions"),
            "https://integrate.api.nvidia.com/v1/chat/completions",
        )
        self.assertEqual(
            _openai_compat_url("http://127.0.0.1:1234", "/models"),
            "http://127.0.0.1:1234/v1/models",
        )

    def test_explicit_nim_model_disables_silent_fallback(self):
        self.assertEqual(_nim_model_chain("moonshotai/kimi-k2-instruct"), ["moonshotai/kimi-k2-instruct"])

    def test_default_nim_chain_starts_with_kimi_instruct(self):
        chain = _nim_model_chain(None)
        self.assertGreaterEqual(len(chain), 1)
        self.assertEqual(chain[0], "moonshotai/kimi-k2-instruct")

    def test_rate_limit_snapshot_contains_budget_fields(self):
        snapshot = nim_rate_limit_snapshot()
        self.assertIn("limit_per_minute", snapshot)
        self.assertIn("used_in_current_window", snapshot)
        self.assertIn("remaining_in_current_window", snapshot)


if __name__ == "__main__":
    unittest.main()
