import unittest

import numpy as np

from src.v17.relativistic_cone import RelativisticCone


class V17RelativisticConeTests(unittest.TestCase):
    def test_envelope_has_inner_and_outer_bounds(self) -> None:
        returns = np.array([0.001, -0.0008, 0.0012, -0.0004, 0.0006], dtype=np.float64)
        cone = RelativisticCone(returns).envelope(2300.0, 3, hurst_positive=0.56, hurst_negative=0.63)
        self.assertEqual(len(cone["cone_upper"]), 3)
        self.assertTrue(cone["compact_support"])

    def test_branch_plausibility_penalises_large_moves(self) -> None:
        rc = RelativisticCone(np.array([0.001, 0.0012, -0.0009], dtype=np.float64))
        plausible = rc.branch_plausibility(np.array([0.0005, 0.0004, 0.0003], dtype=np.float64), 3)
        implausible = rc.branch_plausibility(np.array([0.05, 0.05, 0.05], dtype=np.float64), 3)
        self.assertGreater(plausible, implausible)


if __name__ == "__main__":
    unittest.main()
