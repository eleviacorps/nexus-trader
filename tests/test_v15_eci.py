import unittest
from datetime import datetime, timezone
from pathlib import Path

from src.v15.eci import EconomicCalendarIntegration, EconomicEvent


class V15ECITests(unittest.TestCase):
    def test_context_detects_pre_release_and_reaction_windows(self) -> None:
        event_time = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
        eci = EconomicCalendarIntegration([EconomicEvent(timestamp=event_time, event_type='cpi', importance=3)])
        pre = eci.get_context_at(datetime(2024, 1, 1, 11, 30, tzinfo=timezone.utc))
        reaction = eci.get_context_at(datetime(2024, 1, 1, 12, 10, tzinfo=timezone.utc))
        self.assertTrue(pre['pre_release'])
        self.assertTrue(reaction['reaction_window'])

    def test_from_csv_loads_normalized_events(self) -> None:
        scratch = Path('tests/.tmp/test_v15_eci_calendar.csv')
        scratch.parent.mkdir(parents=True, exist_ok=True)
        scratch.write_text('datetime,event_type,importance\n2024-01-01T12:00:00Z,CPI,3\n', encoding='utf-8')
        try:
            eci = EconomicCalendarIntegration.from_csv(scratch)
        finally:
            if scratch.exists():
                scratch.unlink()
        self.assertEqual(len(eci.events), 1)
        self.assertEqual(eci.events[0].event_type, 'cpi')


if __name__ == '__main__':
    unittest.main()
