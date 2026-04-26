import unittest

import torch

from src.v17.lee_coc import LeeCOC


class V17LeeCOCTests(unittest.TestCase):
    def test_forward_preserves_shape(self) -> None:
        activation = LeeCOC()
        x = torch.randn(4, 8)
        y = activation(x)
        self.assertEqual(tuple(y.shape), tuple(x.shape))


if __name__ == "__main__":
    unittest.main()
