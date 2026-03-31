import unittest

import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

from src.pipeline.persona import _crowd_bias_from_embeddings, _news_bias_from_embeddings, align_macro_to_price


@unittest.skipIf(pd is None, "pandas is required for persona pipeline tests")
class PersonaPipelineTests(unittest.TestCase):
    def test_align_macro_to_price_produces_expected_columns(self):
        price_index = pd.date_range("2026-01-01", periods=4, freq="h")
        macro = pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-12-31", "2026-01-01"]),
                "DTWEXBGS": [120.0, 118.0],
                "UUP": [28.0, 27.5],
                "DFII10": [2.2, 2.0],
                "DGS10": [4.6, 4.4],
                "VIXCLS": [16.0, 20.0],
                "T10YIE": [2.3, 2.4],
                "TLT": [88.0, 89.5],
                "DCOILWTICO": [72.0, 74.0],
                "GC_F": [2650.0, 2675.0],
                "GLD": [245.0, 247.0],
            }
        )
        aligned = align_macro_to_price(price_index, macro)
        self.assertEqual(len(aligned), 4)
        self.assertIn("macro_bias", aligned.columns)
        self.assertIn("macro_shock", aligned.columns)
        self.assertIn("macro_driver", aligned.columns)

    def test_news_bias_helper_respects_width(self):
        news = np.array([[0.0] * 32, [2.0, 1.0, -1.0] + [0.0] * 29], dtype=np.float32)
        bias, intensity = _news_bias_from_embeddings(news)
        self.assertEqual(bias.shape, (2,))
        self.assertEqual(intensity.shape, (2,))
        self.assertGreater(intensity[1], intensity[0])

    def test_crowd_bias_helper_respects_width(self):
        crowd = np.zeros((2, 32), dtype=np.float32)
        crowd[0, 0] = 0.8
        crowd[1, 0] = 0.2
        crowd[1, 15] = 1.0
        bias, extreme = _crowd_bias_from_embeddings(crowd)
        self.assertEqual(bias.shape, (2,))
        self.assertEqual(extreme.shape, (2,))
        self.assertGreater(extreme[1], extreme[0])


if __name__ == "__main__":
    unittest.main()
