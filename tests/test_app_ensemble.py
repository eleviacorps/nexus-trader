import unittest

from src.service.app import build_ensemble_prediction


class EnsemblePredictionTests(unittest.TestCase):
    def test_ensemble_keeps_branch_signal_primary(self):
        payload = {
            "simulation": {
                "mean_probability": 0.64,
                "consensus_score": 0.82,
                "overall_confidence": 0.71,
            },
            "current_row": {
                "quant_route_prob_up": 0.68,
                "quant_route_prob_down": 0.18,
                "quant_route_confidence": 0.62,
                "quant_transition_risk": 0.21,
                "macro_shock": 0.12,
                "macro_bias": 0.19,
            },
            "bot_swarm": {
                "aggregate": {
                    "bullish_probability": 0.61,
                    "confidence": 0.67,
                    "regime_affinity": {
                        "trend": 0.74,
                        "reversal": 0.16,
                        "macro_shock": 0.22,
                        "balanced": 0.18,
                    },
                }
            },
            "llm_context": {
                "content": {
                    "institutional_bias": -0.2,
                    "whale_bias": -0.1,
                    "retail_bias": 0.1,
                }
            },
        }
        model_prediction = {
            "bullish_probability": 0.58,
            "model_diagnostics": {
                "regime_probabilities": {
                    "trend": 0.62,
                    "reversal": 0.12,
                    "macro_shock": 0.08,
                    "balanced": 0.18,
                }
            },
        }
        prediction = build_ensemble_prediction(payload, model_prediction)
        self.assertEqual(prediction["signal"], "bullish")
        self.assertGreater(prediction["weights"]["branch"], prediction["weights"]["llm"])
        self.assertGreaterEqual(prediction["confidence"], 0.0)
        self.assertLessEqual(prediction["confidence"], 1.0)
        self.assertIn("regime_mix", prediction)
        self.assertAlmostEqual(sum(prediction["weights"].values()), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
