import unittest

from src.service.llm_sidecar import get_nim_model, parse_kimi_response
from src.v18.kimi_system_prompt import KIMI_SYSTEM_PROMPT, build_kimi_user_message


class V18KimiPromptTests(unittest.TestCase):
    def test_system_prompt_mentions_json_contract(self) -> None:
        self.assertIn('"stance"', KIMI_SYSTEM_PROMPT)
        self.assertIn("NUMERIC GLOSSARY", KIMI_SYSTEM_PROMPT)

    def test_build_user_message_includes_market_sections(self) -> None:
        message = build_kimi_user_message(
            {
                "market": {"current_price": 2350.25},
                "simulation": {
                    "scenario_bias": "bullish",
                    "overall_confidence": 0.64,
                    "cabr_score": 0.61,
                    "cpm_score": 0.72,
                    "cone_width_pips": 142.0,
                    "hurst_overall": 0.58,
                    "hurst_asymmetry": 0.11,
                    "testosterone_index": {"retail": 0.32, "institutional": 0.18},
                },
                "technical_analysis": {
                    "structure": "bullish",
                    "location": "discount",
                    "rsi_14": 57.0,
                    "atr_14": 18.2,
                    "equilibrium": 2346.8,
                    "nearest_support": {"price": 2342.0},
                    "nearest_resistance": {"price": 2364.0},
                    "order_blocks": [],
                },
                "bot_swarm": {"aggregate": {"signal": "bullish", "bullish_probability": 0.68, "disagreement": 0.12}},
                "sqt": {"label": "HOT", "rolling_accuracy": 0.7},
                "news_feed": [{"title": "<b>Gold rallies</b> - Reuters", "source": "Reuters", "sentiment": 0.2}],
                "mfg": {"disagreement": 0.0002, "consensus_drift": 0.0001},
            },
            "XAUUSD",
        )
        self.assertIn("SIMULATION SUMMARY", message)
        self.assertIn("Gold rallies", message)
        self.assertNotIn("<b>", message)

    def test_parse_kimi_response_accepts_fenced_json(self) -> None:
        payload = parse_kimi_response(
            """```json
{"stance":"BUY","confidence":"MODERATE","entry_zone":[1,2],"stop_loss":0.5,"take_profit":3,"hold_time":"current_bar","reasoning":"ok","key_risk":"risk","crowd_note":"crowd","regime_note":"regime","invalidation":0.5}
```"""
        )
        self.assertEqual(payload["stance"], "BUY")
        self.assertEqual(payload["entry_zone"], [1, 2])

    def test_parse_kimi_response_falls_back_safely(self) -> None:
        payload = parse_kimi_response("not json")
        self.assertEqual(payload["stance"], "HOLD")
        self.assertTrue(payload["error"])

    def test_get_nim_model_prefers_requested_model(self) -> None:
        self.assertEqual(get_nim_model("meta/llama-3.1-70b-instruct"), "meta/llama-3.1-70b-instruct")
        self.assertEqual(get_nim_model(""), "moonshotai/kimi-k2-5")


if __name__ == "__main__":
    unittest.main()
