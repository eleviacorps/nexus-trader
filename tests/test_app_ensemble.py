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
            "llm_context": {
                "content": {
                    "institutional_bias": -0.2,
                    "whale_bias": -0.1,
                    "retail_bias": 0.1,
                }
            },
        }
        model_prediction = {"bullish_probability": 0.58}
        prediction = build_ensemble_prediction(payload, model_prediction)
        self.assertEqual(prediction["signal"], "bullish")
        self.assertGreater(prediction["weights"]["branch"], prediction["weights"]["llm"])
        self.assertGreaterEqual(prediction["confidence"], 0.0)
        self.assertLessEqual(prediction["confidence"], 1.0)


if __name__ == "__main__":
    unittest.main()
