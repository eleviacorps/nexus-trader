import unittest

from src.utils.training_splits import split_by_years


class TrainingSplitTests(unittest.TestCase):
    def test_split_by_years_builds_expected_slices(self):
        timestamps = [
            '2020-12-31T23:58:00',
            '2021-01-01T00:00:00',
            '2021-06-01T00:00:00',
            '2022-01-01T00:00:00',
            '2022-06-01T00:00:00',
            '2023-01-01T00:00:00',
            '2023-06-01T00:00:00',
        ]
        split = split_by_years(timestamps, sequence_len=2, train_years=[2021], val_years=[2022], test_years=[2023])
        self.assertEqual(split.mode, 'year')
        self.assertGreaterEqual(split.train.start, 0)
        self.assertGreater(split.train.stop, split.train.start)
        self.assertGreater(split.val.stop, split.val.start)
        self.assertGreater(split.test.stop, split.test.start)


if __name__ == '__main__':
    unittest.main()
