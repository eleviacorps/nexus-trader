"""V24 Diffusion v2 (U-Net + epsilon-prediction) tests.

Covers: U-Net architecture, noise scheduler, dataset, generator with CFG,
training loss backward pass, and DDIM/DDPM sampling.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.v24.diffusion.unet_1d import DiffusionUNet1D, ResBlock1d, FiLMConditioning
from src.v24.diffusion.scheduler import NoiseScheduler
from src.v24.diffusion.dataset import DiffusionDataset, DatasetSlice, split_by_year
from src.v24.diffusion.generator import DiffusionPathGeneratorV2, GeneratorConfig


class TestUNet1D(unittest.TestCase):
    def setUp(self):
        self.model = DiffusionUNet1D(
            in_channels=100,
            base_channels=64,
            channel_multipliers=(1, 2, 4),
            num_res_blocks=2,
            ctx_dim=100,
        )
        self.B, self.C, self.L = 2, 100, 64

    def test_output_shape(self):
        x = torch.randn(self.B, self.C, self.L)
        t = torch.randint(0, 1000, (self.B,))
        ctx = torch.randn(self.B, 100)
        out = self.model(x, t, context=ctx)
        self.assertEqual(out.shape, (self.B, self.C, self.L))

    def test_without_context(self):
        x = torch.randn(self.B, self.C, self.L)
        t = torch.randint(0, 1000, (self.B,))
        out = self.model(x, t)
        self.assertEqual(out.shape, (self.B, self.C, self.L))

    def test_various_lengths(self):
        for L in [32, 64, 128]:
            x = torch.randn(1, self.C, L)
            out = self.model(x, torch.randint(0, 1000, (1,)), context=torch.randn(1, 100))
            self.assertEqual(out.shape, (1, self.C, L))

    def test_backward_pass(self):
        x = torch.randn(self.B, self.C, self.L)
        t = torch.randint(0, 1000, (self.B,))
        ctx = torch.randn(self.B, 100)
        out = self.model(x, t, context=ctx)
        loss = out.mean()
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.model.parameters())
        self.assertTrue(has_grad)

    def test_parameter_count(self):
        n = sum(p.numel() for p in self.model.parameters())
        self.assertGreater(n, 100_000)
        self.assertLess(n, 50_000_000)


class TestNoiseScheduler(unittest.TestCase):
    def setUp(self):
        self.scheduler = NoiseScheduler(num_timesteps=1000)

    def test_buffer_shapes(self):
        self.assertEqual(self.scheduler.betas.shape, (1000,))
        self.assertEqual(self.scheduler.alphas_cumprod.shape, (1000,))

    def test_q_sample_shape(self):
        x = torch.randn(4, 100, 64)
        t = torch.randint(0, 1000, (4,))
        x_t = self.scheduler.q_sample(x, t)
        self.assertEqual(x_t.shape, x.shape)

    def test_q_sample_with_noise(self):
        x = torch.randn(4, 100, 64)
        t = torch.randint(0, 1000, (4,))
        noise = torch.randn_like(x)
        x_t = self.scheduler.q_sample(x, t, noise=noise)
        self.assertEqual(x_t.shape, x.shape)

    def test_training_loss_scalar(self):
        model = DiffusionUNet1D(in_channels=10, base_channels=32, channel_multipliers=(1, 2), num_res_blocks=1, ctx_dim=10)
        x = torch.randn(4, 10, 32)
        ctx = torch.randn(4, 10)
        loss = self.scheduler.training_loss(model, x, ctx)
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertTrue(loss.item() > 0)

    def test_training_loss_backward(self):
        model = DiffusionUNet1D(in_channels=10, base_channels=32, channel_multipliers=(1, 2), num_res_blocks=1, ctx_dim=10)
        x = torch.randn(4, 10, 32)
        ctx = torch.randn(4, 10)
        loss = self.scheduler.training_loss(model, x, ctx)
        loss.backward()

    def test_beta_range(self):
        self.assertTrue(self.scheduler.betas.min() >= 0.0001)
        self.assertTrue(self.scheduler.betas.max() <= 0.9999)

    def test_alphas_cumprod_decreasing(self):
        ac = self.scheduler.alphas_cumprod
        self.assertTrue(torch.all(ac[1:] <= ac[:-1] + 1e-6))


class TestSchedulerSampling(unittest.TestCase):
    def setUp(self):
        self.model = DiffusionUNet1D(in_channels=10, base_channels=32, channel_multipliers=(1, 2), num_res_blocks=1, ctx_dim=10)
        self.scheduler = NoiseScheduler(num_timesteps=100)
        self.model.eval()

    def test_ddpm_sample(self):
        out = self.scheduler.ddpm_sample(self.model, shape=(1, 10, 32), context=torch.randn(1, 10), device=torch.device("cpu"))
        self.assertEqual(out.shape, (1, 10, 32))

    def test_ddim_sample(self):
        out = self.scheduler.ddim_sample(self.model, shape=(1, 10, 32), context=torch.randn(1, 10), num_steps=5, device=torch.device("cpu"))
        self.assertEqual(out.shape, (1, 10, 32))

    def test_p_sample(self):
        x_t = torch.randn(1, 10, 32)
        t = torch.tensor([50])
        ctx = torch.randn(1, 10)
        out = self.scheduler.p_sample(self.model, x_t, t, ctx)
        self.assertEqual(out.shape, (1, 10, 32))


class TestDataset(unittest.TestCase):
    def setUp(self):
        self.fused_path = PROJECT_ROOT / "data" / "features" / "diffusion_fused_405k.npy"
        self.skip_if_no_data()

    def skip_if_no_data(self):
        if not self.fused_path.exists():
            self.skipTest("diffusion_fused_405k.npy not found")

    def test_dataset_loads(self):
        ds = DiffusionDataset(self.fused_path, sequence_len=120, row_slice=DatasetSlice(0, 100))
        self.assertGreater(len(ds), 0)

    def test_dataset_shapes(self):
        ds = DiffusionDataset(self.fused_path, sequence_len=120, row_slice=DatasetSlice(0, 100))
        w, c = ds[0]
        self.assertEqual(w.shape, (100, 120))
        self.assertEqual(c.shape, (100,))

    def test_dataset_slice_range(self):
        ds = DiffusionDataset(self.fused_path, sequence_len=120, row_slice=DatasetSlice(10, 50))
        self.assertEqual(len(ds), 40)

    def test_split_by_year(self):
        timestamps = np.array([f"{y}-01-01" for y in range(2009, 2027)] * 22500, dtype=object)
        total = len(timestamps)
        train_s, val_s, test_s = split_by_year(total, 10, timestamps=timestamps)
        self.assertGreater(len(train_s), 0)
        self.assertGreater(len(val_s), 0)
        self.assertGreater(len(test_s), 0)
        self.assertGreater(val_s.start, train_s.start)
        self.assertGreater(test_s.start, val_s.start)

    def test_split_by_year_fallback(self):
        train_s, val_s, test_s = split_by_year(10000, 10)
        self.assertAlmostEqual(len(train_s) / 9990, 0.70, places=1)
        self.assertAlmostEqual(len(val_s) / 9990, 0.15, places=1)


class TestGenerator(unittest.TestCase):
    def setUp(self):
        self.config = GeneratorConfig(
            in_channels=10,
            sequence_length=32,
            base_channels=32,
            channel_multipliers=(1, 2),
            num_timesteps=100,
            ctx_dim=10,
            guidance_scale=3.0,
            num_paths=2,
            sampling_steps=5,
        )
        self.model = DiffusionUNet1D(
            in_channels=10, base_channels=32, channel_multipliers=(1, 2),
            num_res_blocks=1, ctx_dim=10,
        )
        self.scheduler = NoiseScheduler(num_timesteps=100)
        self.gen = DiffusionPathGeneratorV2(
            config=self.config, model=self.model, scheduler=self.scheduler, device="cpu",
        )

    def test_generate_paths_count(self):
        ws = {"return_1": 0.01, "vol": 0.15}
        paths = self.gen.generate_paths(ws, num_paths=3, steps=5)
        self.assertEqual(len(paths), 3)

    def test_path_has_required_keys(self):
        ws = {"return_1": 0.01}
        paths = self.gen.generate_paths(ws, num_paths=2, steps=5)
        for p in paths:
            self.assertIn("path_id", p)
            self.assertIn("data", p)
            self.assertIn("confidence", p)
            self.assertIn("metadata", p)

    def test_confidence_range(self):
        ws = {"return_1": 0.01}
        paths = self.gen.generate_paths(ws, num_paths=4, steps=5)
        for p in paths:
            self.assertGreaterEqual(p["confidence"], 0.1)
            self.assertLessEqual(p["confidence"], 0.99)

    def test_cfg_prediction(self):
        x = torch.randn(2, 10, 32)
        t = torch.randint(0, 100, (2,))
        ctx = torch.randn(2, 10)
        self.gen.model.eval()
        eps = self.gen._cfg_model_forward(x, t, ctx)
        self.assertEqual(eps.shape, (2, 10, 32))

    def test_learned_confidence(self):
        paths = np.random.randn(8, 32, 10).astype(np.float32)
        conf = DiffusionPathGeneratorV2._compute_learned_confidence(paths)
        self.assertEqual(conf.shape, (8,))
        self.assertTrue(np.all(conf >= 0.1))
        self.assertTrue(np.all(conf <= 0.99))


def run_all_tests():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [TestUNet1D, TestNoiseScheduler, TestSchedulerSampling, TestDataset, TestGenerator]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


if __name__ == "__main__":
    run_all_tests()
