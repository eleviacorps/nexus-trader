from __future__ import annotations

import unittest

from src.service.llm_sidecar import build_swarm_judgment_prompt


class LlmSwarmPromptTests(unittest.TestCase):
    def test_build_swarm_judgment_prompt_contains_schema_fields(self):
        prompt = build_swarm_judgment_prompt("XAUUSD", {"simulation": {"scenario_bias": "bullish"}})
        self.assertIn("master_bias", prompt)
        self.assertIn("crowd_emotion", prompt)
        self.assertIn("minority_case", prompt)
        self.assertIn("manual_stance", prompt)
        self.assertIn("public_reaction_lines", prompt)


if __name__ == "__main__":
    unittest.main()
