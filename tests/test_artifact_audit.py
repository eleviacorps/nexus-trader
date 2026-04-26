import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.backtest.artifact_audit import audit_model_artifacts


class ArtifactAuditTests(unittest.TestCase):
    def test_audit_model_artifacts_reports_findings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            training_summary = root / "training_summary.json"
            walkforward_report = root / "walkforward_report.json"
            manifest = root / "model_manifest.json"
            precision_gate = root / "precision_gate.json"
            gate_context = root / "gate_context.npy"
            timestamps = root / "timestamps.npy"
            targets = root / "targets_multihorizon.npz"

            training_summary.write_text(
                json.dumps(
                    {
                        "test_metrics": {"primary": {"roc_auc": 0.51}},
                        "val_metrics": {"primary": {"roc_auc": 0.60}},
                    }
                ),
                encoding="utf-8",
            )
            walkforward_report.write_text(
                json.dumps(
                    {
                        "overall": {
                            "calibrated_metrics": {"roc_auc": 0.50},
                            "gate_positive_rate": 0.01,
                        }
                    }
                ),
                encoding="utf-8",
            )
            manifest.write_text(json.dumps({"run_tag": "unit", "checkpoint_path": "final.ckpt"}), encoding="utf-8")
            precision_gate.write_text(json.dumps({"train_accuracy": 0.95, "train_precision": 0.10}), encoding="utf-8")
            np.save(gate_context, np.zeros((5, 3), dtype=np.float32))
            np.save(timestamps, np.arange(4))
            np.savez(targets, target_5m=np.zeros(4, dtype=np.float32), target_10m=np.zeros(4, dtype=np.float32))

            report = audit_model_artifacts(
                training_summary_path=training_summary,
                walkforward_report_path=walkforward_report,
                manifest_path=manifest,
                precision_gate_path=precision_gate,
                gate_context_path=gate_context,
                timestamps_path=timestamps,
                targets_multihorizon_path=targets,
            )
            self.assertIn("precision_gate_has_high_train_accuracy_but_low_train_precision", report["findings"])
            self.assertIn("validation_test_roc_gap_is_large", report["findings"])
            self.assertIn("gate_participation_near_zero", report["findings"])
            self.assertIn("gate_context_length_differs_from_timestamp_length", report["findings"])


if __name__ == "__main__":
    unittest.main()
