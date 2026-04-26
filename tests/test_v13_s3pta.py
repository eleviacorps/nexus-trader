import unittest
from pathlib import Path

from src.v13.s3pta import PaperTradeAccumulator


class S3PTATests(unittest.TestCase):
    def test_completed_trade_summary(self) -> None:
        path = Path('outputs/v13/test_paper_trades.jsonl')
        if path.exists():
            path.unlink()
        acc = PaperTradeAccumulator(path)
        acc.log_completed_trade(
            symbol='XAUUSD',
            direction='BUY',
            uts_score=0.7,
            cabr_score=0.65,
            regime='trending_up',
            entry_price=2000.0,
            exit_price=2001.0,
            entry_time='2024-01-01T00:00:00+00:00',
            exit_time='2024-01-01T00:15:00+00:00',
        )
        summary = acc.summary()
        self.assertEqual(summary['count'], 1)
        self.assertEqual(summary['pending'], 0)
        path.unlink(missing_ok=True)


if __name__ == '__main__':
    unittest.main()
