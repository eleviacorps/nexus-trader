import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.v12 import (
    LiveConfidenceCalibrator,
    compute_bar_consistent_features,
    compute_online_feature_frame,
    run_feature_consistency_audit,
)
from src.v12.sarv import run_sarv_validation
from src.v12.tctl import score_tctl_model, train_tctl_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints' / 'v12'
TEST_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


class V12FeatureConsistencyTests(unittest.TestCase):
    def _bars(self, count: int = 320) -> pd.DataFrame:
        index = pd.date_range('2024-01-01 00:00:00+00:00', periods=count, freq='min', tz='UTC')
        base = 2000.0 + np.linspace(0.0, 6.0, count) + 0.8 * np.sin(np.arange(count) / 9.0)
        close = base + 0.15 * np.cos(np.arange(count) / 7.0)
        open_ = base - 0.10 * np.sin(np.arange(count) / 5.0)
        high = np.maximum(open_, close) + 0.18
        low = np.minimum(open_, close) - 0.18
        volume = 100.0 + (np.arange(count) % 17).astype(float)
        return pd.DataFrame(
            {
                'open': open_,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume,
            },
            index=index,
        )

    def _synthetic_tctl_frame(self, count: int = 48) -> pd.DataFrame:
        timestamps = pd.date_range('2024-02-01 00:00:00+00:00', periods=count, freq='5min', tz='UTC')
        signal_a = np.linspace(-1.0, 1.0, count)
        signal_b = np.sin(np.linspace(0.0, 3.0 * np.pi, count))
        pnl = 0.25 * signal_a + 0.10 * signal_b
        pnl[::7] *= -0.5
        stage_name = np.where((np.arange(count) % 3) == 0, 'open', np.where((np.arange(count) % 3) == 1, 'pcop_5m', 'pcop_10m'))
        return pd.DataFrame(
            {
                'timestamp': timestamps,
                'sample_id': np.arange(count, dtype=np.int64),
                'setl_target_net_unit_pnl': pnl.astype(np.float32),
                'signal_a': signal_a.astype(np.float32),
                'signal_b': signal_b.astype(np.float32),
                'stage_name': stage_name,
                'dominant_regime': np.where(signal_a > 0.25, 'bull_trend', 'range'),
            }
        )

    def _checkpoint_path(self, name: str) -> Path:
        path = TEST_CHECKPOINT_DIR / name
        if path.exists():
            path.unlink()
        return path

    def test_online_engine_matches_bar_consistent_archive(self) -> None:
        bars = self._bars()
        archive = compute_bar_consistent_features(bars)
        live = compute_online_feature_frame(bars, warmup_bars=60)
        archive = archive.loc[live.index, live.columns]
        for column in live.columns:
            left = archive[column].to_numpy(dtype=np.float64)
            right = live[column].to_numpy(dtype=np.float64)
            if np.std(left) <= 1e-12 and np.std(right) <= 1e-12:
                self.assertTrue(np.allclose(left, right))
                continue
            corr = np.corrcoef(left, right)[0, 1]
            self.assertGreaterEqual(float(corr), 0.999)

    def test_audit_flags_injected_drift(self) -> None:
        bars = self._bars()
        archive = compute_bar_consistent_features(bars)
        archive['ema_50_ratio'] = archive['ema_50_ratio'] + np.linspace(-0.5, 0.5, len(archive))
        report = run_feature_consistency_audit(
            raw_bars=bars,
            archive_features=archive,
            warmup_bars=60,
            pass_threshold=0.95,
        )
        self.assertIn('ema_50_ratio', report['legacy_archive_vs_live']['fail_features'])
        self.assertEqual(len(report['bcfe_self_check']['fail_features']), 0)

    def test_tctl_does_not_collapse(self) -> None:
        frame = self._synthetic_tctl_frame()
        checkpoint = self._checkpoint_path('test_unit_tctl.pt')
        result = train_tctl_model(
            frame,
            feature_names=('signal_a', 'signal_b'),
            device='cpu',
            epochs=3,
            batch_size=16,
            checkpoint_path=checkpoint,
        )
        scores = score_tctl_model(result['model'], frame.iloc[:20], feature_names=('signal_a', 'signal_b'), device='cpu')
        self.assertGreaterEqual(len(set(np.round(scores, 4).tolist())), 3)

    def test_sarv_stage2_gap_within_bounds_on_identical_replay(self) -> None:
        frame = self._synthetic_tctl_frame()
        train_frame = frame.iloc[:24].reset_index(drop=True)
        valid_frame = frame.iloc[24:].reset_index(drop=True)
        checkpoint = self._checkpoint_path('test_unit_sarv.pt')
        result = train_tctl_model(
            train_frame,
            feature_names=('signal_a', 'signal_b'),
            device='cpu',
            epochs=3,
            batch_size=16,
            checkpoint_path=checkpoint,
        )
        report = run_sarv_validation(
            model=result['model'],
            train_candidates=train_frame,
            archive_candidates=valid_frame,
            bar_replay_candidates=valid_frame.copy(),
            feature_names=('signal_a', 'signal_b'),
        )
        self.assertLessEqual(float(report['stage_2']['gap_vs_stage1_winrate'] or 0.0), 0.05)

    def test_live_confidence_calibrator_smoke(self) -> None:
        calibrator = LiveConfidenceCalibrator(decay_factor=0.99, min_samples=10)
        for raw_score, won in [
            (0.10, False),
            (0.15, False),
            (0.20, False),
            (0.30, False),
            (0.45, False),
            (0.55, True),
            (0.60, True),
            (0.70, True),
            (0.80, True),
            (0.90, True),
            (0.95, True),
            (0.85, True),
        ]:
            calibrator.record_outcome(raw_score, won)
        calibrated = calibrator.calibrate(0.75)
        self.assertGreater(calibrated, 0.5)
        self.assertLess(calibrator.calibration_error(), 0.35)


if __name__ == '__main__':
    unittest.main()
