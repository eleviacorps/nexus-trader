from __future__ import annotations

import unittest

from src.v14.acm import AsymmetricMemory, build_acm_memories, fear_indices_from_closes


class TestV14ACM(unittest.TestCase):
    def test_memory_penalizes_losses_more_than_gains(self) -> None:
        mem = AsymmetricMemory(persona_name="retail", loss_weight=2.0)
        mem.update(-0.01)
        loss_fear = mem.fear_index
        mem = AsymmetricMemory(persona_name="retail", loss_weight=2.0)
        mem.update(0.01)
        gain_fear = mem.fear_index
        self.assertGreater(loss_fear, gain_fear)

    def test_build_memories_smoke(self) -> None:
        seq = [
            {"close": 100.0},
            {"close": 99.0},
            {"close": 99.5},
            {"close": 98.5},
        ]
        memories = build_acm_memories(seq)
        self.assertIn("retail", memories)
        self.assertGreaterEqual(memories["retail"].fear_index, 0.0)

    def test_fear_indices_from_closes(self) -> None:
        frame = fear_indices_from_closes([100.0, 99.0, 101.0, 100.5])
        self.assertIn("fear_index_retail", frame.columns)
        self.assertEqual(len(frame), 4)


if __name__ == "__main__":
    unittest.main()
