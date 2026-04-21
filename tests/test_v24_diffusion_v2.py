"""V24 Diffusion v2 (U-Net + epsilon-prediction) tests.

Phase 0.5: Adds tests for temporal encoder, temporal FiLM in ResBlocks,
temporal cross-attention in decoder, and dataset with context_len.
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
from src.v24.diffusion.temporal_encoder import TemporalEncoder


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


class TestUNet1DTemporal(unittest.TestCase):
    def setUp(self):
        self.temporal_dim = 256
        self.d_gru = 256
        self.model = DiffusionUNet1D(
            in_channels=144,
            base_channels=64,
            channel_multipliers=(1, 2, 4),
            num_res_blocks=2,
            ctx_dim=144,
            temporal_dim=self.temporal_dim,
            d_gru=self.d_gru,
        )
        self.B, self.C, self.L = 2, 144, 64
        self.T_past = 32

    def test_temporal_output_shape(self):
        x = torch.randn(self.B, self.C, self.L)
        t = torch.randint(0, 1000, (self.B,))
        ctx = torch.randn(self.B, 144)
        temporal_seq = torch.randn(self.B, self.T_past, self.d_gru)
        temporal_emb = torch.randn(self.B, self.temporal_dim)
        out = self.model(x, t, context=ctx, temporal_seq=temporal_seq, temporal_emb=temporal_emb)
        self.assertEqual(out.shape, (self.B, self.C, self.L))

    def test_temporal_backward_pass(self):
        x = torch.randn(self.B, self.C, self.L)
        t = torch.randint(0, 1000, (self.B,))
        ctx = torch.randn(self.B, 144)
        temporal_seq = torch.randn(self.B, self.T_past, self.d_gru)
        temporal_emb = torch.randn(self.B, self.temporal_dim)
        out = self.model(x, t, context=ctx, temporal_seq=temporal_seq, temporal_emb=temporal_emb)
        loss = out.mean()
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.model.parameters())
        self.assertTrue(has_grad)

    def test_temporal_decoder_cross_attns_exist(self):
        self.assertTrue(hasattr(self.model, 'decoder_temporal_cross_attns'))
        self.assertEqual(len(self.model.decoder_temporal_cross_attns), 3)
        for level_attns in self.model.decoder_temporal_cross_attns:
            for attn in level_attns:
                self.assertIsNotNone(attn)

    def test_resblock_temporal_film(self):
        block = ResBlock1d(64, 64, time_dim=256, temporal_dim=256)
        x = torch.randn(2, 64, 32)
        t_emb = torch.randn(2, 256)
        temporal_emb = torch.randn(2, 256)
        out = block(x, t_emb, temporal_emb)
        self.assertEqual(out.shape, (2, 64, 32))

    def test_without_temporal_backward_compat(self):
        x = torch.randn(self.B, self.C, self.L)
        t = torch.randint(0, 1000, (self.B,))
        ctx = torch.randn(self.B, 144)
        out = self.model(x, t, context=ctx)
        self.assertEqual(out.shape, (self.B, self.C, self.L))

    def test_temporal_parameter_count(self):
        n = sum(p.numel() for p in self.model.parameters())
        self.assertGreater(n, 100_000)


class TestTemporalEncoder(unittest.TestCase):
    def setUp(self):
        self.encoder = TemporalEncoder(
            in_features=144,
            d_gru=256,
            num_layers=2,
            film_dim=256,
        )
        self.B, self.T_past, self.C = 2, 32, 144

    def test_output_shapes(self):
        past = torch.randn(self.B, self.T_past, self.C)
        hidden_seq, film_emb, final_hidden = self.encoder(past)
        self.assertEqual(hidden_seq.shape, (self.B, self.T_past, 256))
        self.assertEqual(film_emb.shape, (self.B, 256))
        self.assertEqual(final_hidden.shape, (self.B, 256))

    def test_backward_pass(self):
        past = torch.randn(self.B, self.T_past, self.C)
        hidden_seq, film_emb, final_hidden = self.encoder(past)
        loss = film_emb.mean() + hidden_seq.mean()
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.encoder.parameters())
        self.assertTrue(has_grad)

    def test_parameter_count(self):
        n = sum(p.numel() for p in self.encoder.parameters())
        self.assertGreater(n, 100_000)
        self.assertLess(n, 10_000_000)

    def test_deterministic_with_same_input(self):
        past = torch.randn(self.B, self.T_past, self.C)
        self.encoder.eval()
        with torch.no_grad():
            h1, f1, _ = self.encoder(past)
            h2, f2, _ = self.encoder(past)
        self.assertTrue(torch.allclose(h1, h2, atol=1e-5))
        self.assertTrue(torch.allclose(f1, f2, atol=1e-5))


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

    def test_training_loss_with_temporal(self):
        model = DiffusionUNet1D(in_channels=10, base_channels=32, channel_multipliers=(1, 2), num_res_blocks=1, ctx_dim=10, temporal_dim=256, d_gru=64)
        x = torch.randn(4, 10, 32)
        ctx = torch.randn(4, 10)
        temporal_seq = torch.randn(4, 16, 64)
        temporal_emb = torch.randn(4, 256)
        loss = self.scheduler.training_loss(model, x, ctx, temporal_seq=temporal_seq, temporal_emb=temporal_emb)
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertTrue(loss.item() > 0)

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

    def test_ddim_with_temporal(self):
        model = DiffusionUNet1D(in_channels=10, base_channels=32, channel_multipliers=(1, 2), num_res_blocks=1, ctx_dim=10, temporal_dim=256, d_gru=64)
        model.eval()
        scheduler = NoiseScheduler(num_timesteps=100)
        temporal_seq = torch.randn(1, 16, 64)
        temporal_emb = torch.randn(1, 256)
        out = scheduler.ddim_sample(model, shape=(1, 10, 32), context=torch.randn(1, 10), num_steps=5, device=torch.device("cpu"),
                                     temporal_seq=temporal_seq, temporal_emb=temporal_emb)
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

    def test_dataset_with_context_len(self):
        ds = DiffusionDataset(self.fused_path, sequence_len=120, row_slice=DatasetSlice(0, 500), context_len=64)
        w, pc = ds[0]
        self.assertEqual(w.shape, (100, 120))
        self.assertEqual(pc.shape, (64, 100))

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
            temporal_gru_dim=0,
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


class TestGeneratorTemporal(unittest.TestCase):
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
            temporal_gru_dim=64,
            context_len=16,
        )
        self.model = DiffusionUNet1D(
            in_channels=10, base_channels=32, channel_multipliers=(1, 2),
            num_res_blocks=1, ctx_dim=10, temporal_dim=256, d_gru=64,
        )
        self.scheduler = NoiseScheduler(num_timesteps=100)
        self.temporal_encoder = TemporalEncoder(
            in_features=10, d_gru=64, num_layers=2, film_dim=256,
        )
        self.gen = DiffusionPathGeneratorV2(
            config=self.config, model=self.model, scheduler=self.scheduler,
            temporal_encoder=self.temporal_encoder, device="cpu",
        )

    def test_generate_paths_with_temporal(self):
        ws = {"return_1": 0.01}
        past_ctx = torch.randn(1, 16, 10)
        paths = self.gen.generate_paths(ws, num_paths=2, steps=5, past_context=past_ctx)
        self.assertEqual(len(paths), 2)

    def test_cfg_with_temporal(self):
        x = torch.randn(2, 10, 32)
        t = torch.randint(0, 100, (2,))
        ctx = torch.randn(2, 10)
        temporal_seq = torch.randn(2, 16, 64)
        temporal_emb = torch.randn(2, 256)
        eps = self.gen._cfg_model_forward(x, t, ctx, temporal_seq, temporal_emb)
        self.assertEqual(eps.shape, (2, 10, 32))

    def test_denormalize_raises_without_stats(self):
        with self.assertRaises(ValueError):
            self.gen.denormalize(np.zeros((2, 32, 10)))


def run_all_tests():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [
        TestUNet1D, TestUNet1DTemporal, TestTemporalEncoder,
        TestNoiseScheduler, TestSchedulerSampling,
        TestDataset, TestGenerator, TestGeneratorTemporal,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


if __name__ == "__main__":
    run_all_tests()
