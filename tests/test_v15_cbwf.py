import json
import unittest
from pathlib import Path

from src.v14.rsc import RegimeStratifiedCalibrator
from src.v15.cbwf import bootstrap_from_walkforward


class V15CBWFTests(unittest.TestCase):
    def test_bootstrap_loads_v14_style_trade_logs(self) -> None:
        payload = {
            'months': [
                {
                    'trade_log': [
                        {'dominant_regime': 'trending_up', 'uts_score': 0.72, 'net_pnl_usd': 12.0},
                        {'dominant_regime': 'ranging', 'cabr_score': 0.44, 'net_pnl_usd': -5.0},
                    ]
                }
            ]
        }
        scratch = Path('tests/.tmp/test_v15_cbwf_walk.json')
        scratch.parent.mkdir(parents=True, exist_ok=True)
        scratch.write_text(json.dumps(payload), encoding='utf-8')
        try:
            calibrator, total = bootstrap_from_walkforward([scratch], RegimeStratifiedCalibrator())
        finally:
            if scratch.exists():
                scratch.unlink()
        self.assertEqual(total, 2)
        self.assertEqual(calibrator._counts['trending_up'], 1)
        self.assertEqual(calibrator._counts['ranging'], 1)


if __name__ == '__main__':
    unittest.main()
