import unittest

from src.v15.participation_audit import audit_walkforward_report


class V15ParticipationAuditTests(unittest.TestCase):
    def test_audit_identifies_primary_bottleneck(self) -> None:
        payload = {
            'version': 'v14',
            'aggregate_trades': 2,
            'months': [
                {
                    'month': '2020-12',
                    'trades_executed': 2,
                    'skip_reason_breakdown': {
                        'lrtd_suppressed': 5,
                        'uts_below_threshold': 1,
                    },
                },
                {
                    'month': '2021-01',
                    'trades_executed': 0,
                    'skip_reason_breakdown': {
                        'lrtd_suppressed': 3,
                    },
                },
            ],
        }
        report = audit_walkforward_report(payload)
        self.assertEqual(report['zero_trade_month_count'], 1)
        self.assertEqual(report['primary_bottleneck']['gate'], 'lrtd_stability')
        self.assertEqual(report['aggregate_trades'], 2)


if __name__ == '__main__':
    unittest.main()
