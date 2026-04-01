from __future__ import annotations

import unittest

from src.service.llm_sidecar import parse_json_text


class LlmSidecarTests(unittest.TestCase):
    def test_parse_json_text_accepts_plain_json(self):
        payload = parse_json_text('{"macro_thesis":"steady","event_severity":0.2}')
        self.assertEqual(payload["macro_thesis"], "steady")
        self.assertEqual(payload["event_severity"], 0.2)

    def test_parse_json_text_accepts_fenced_json(self):
        payload = parse_json_text("```json\n{\"dominant_narrative\":\"dollar firming\"}\n```")
        self.assertEqual(payload["dominant_narrative"], "dollar firming")


if __name__ == "__main__":
    unittest.main()
