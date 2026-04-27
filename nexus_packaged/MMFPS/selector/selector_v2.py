"""Selector v2 - diffusion-based path scorer.

Architecture order:
1) Context encoder (Transformer + GRU + xLSTM-style memory)
2) Path encoder (CNN + GRU)
3) Quant feature block
4) Fusion layer
5) Basic scorer
6) Verify training
7) Diffusion refinement
8) HMM regime features
9) XGBoost path signal
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib

try:
    import xgboost as xgb
except Exception:
    xgb = None


def to_returns(x: torch.Tensor) -> torch.Tensor:
    """Convert level paths to return series.

    x: (..., H)
    returns: (..., H-1)
    """
    prev = x[..., :-1]
    nxt = x[..., 1:]
    return (nxt - prev) / (prev.abs() + 1e-6)


class XLSTMBlock(nn.Module):
    """Lightweight xLSTM-style memory branch for long-horizon context."""

    def __init__(self, d_model: int):
        super().__init__()
        self.lstm = nn.LSTM(d_model, d_model, batch_first=True)
        self.exp_gate = nn.Linear(d_model, d_model)
        self.mem_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.lstm(x)
        gate = torch.sigmoid(self.exp_gate(y))
        mem = torch.tanh(self.mem_proj(y))
        return self.norm(y + gate * mem)


class ContextEncoder(nn.Module):
    """Multi-scale context encoder (Transformer + GRU + xLSTM)."""

    def __init__(
        self,
        feature_dim: int = 144,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
    ):
        super().__init__()
        self.feature_proj = nn.Linear(feature_dim, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

        self.gru = nn.GRU(d_model, d_model, batch_first=True, bidirectional=True)
        self.gru_proj = nn.Linear(d_model * 2, d_model)

        self.xlstm = XLSTMBlock(d_model)

        self.output = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """context: (B, 120, F) -> (B, D)"""
        x = self.feature_proj(context)

        tr = self.transformer(x)
        tr_last = tr[:, -1, :]

        gru_out, _ = self.gru(tr)
        gru_last = self.gru_proj(gru_out[:, -1, :])

        xl = self.xlstm(tr)
        xl_last = xl[:, -1, :]

        fused = torch.cat([tr_last, gru_last, xl_last], dim=-1)
        return self.output(fused)


class PathEncoder(nn.Module):
    """Per-path encoder: 1D CNN + GRU."""

    def __init__(self, path_len: int = 20, d_model: int = 256):
        super().__init__()
        self.path_len = path_len

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
        )

        self.gru = nn.GRU(64, d_model, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(d_model * 2, d_model)

    def forward(self, paths: torch.Tensor) -> torch.Tensor:
        """paths: (B, 128, H) -> (B, 128, D)"""
        bsz, n_paths, horizon = paths.shape
        x = paths.reshape(bsz * n_paths, 1, horizon)
        x = self.cnn(x)
        x = x.transpose(1, 2).contiguous()
        _, h = self.gru(x)
        h = h.transpose(0, 1).reshape(bsz * n_paths, -1)
        x = self.proj(h)
        return x.reshape(bsz, n_paths, -1)


class QuantFeatures(nn.Module):
    """Mandatory per-path quant features.

    Output fields:
    - return mean
    - volatility
    - max drawdown
    - trend strength
    - range
    """

    def __init__(self, path_len: int = 20):
        super().__init__()
        self.path_len = path_len

    def forward(self, paths: torch.Tensor) -> torch.Tensor:
        """paths: (B, 128, H) -> (B, 128, 5)"""
        ret = to_returns(paths)
        ret_mean = ret.mean(dim=-1, keepdim=True)
        volatility = ret.std(dim=-1, keepdim=True).clamp(min=1e-6)
        max_dd = self._max_drawdown(ret)
        trend_strength = ret_mean / volatility

        prange = (paths.max(dim=-1, keepdim=True).values - paths.min(dim=-1, keepdim=True).values)
        prange = prange / (paths[..., 0:1].abs() + 1e-6)

        return torch.cat([ret_mean, volatility, max_dd, trend_strength, prange], dim=-1)

    @staticmethod
    def _max_drawdown(returns: torch.Tensor) -> torch.Tensor:
        cumulative = returns.cumsum(dim=-1)
        running_max = cumulative.cummax(dim=-1).values
        drawdown = cumulative - running_max
        return drawdown.min(dim=-1, keepdim=True).values


class HMMRegimeModel(nn.Module):
    """Loads an offline HMM and returns regime probabilities (B, R).

    If no model is available, returns uniform probabilities.
    """

    def __init__(
        self,
        num_regimes: int = 6,
        model_path: Optional[str] = None,
        enabled: bool = True,
    ):
        super().__init__()
        self.num_regimes = int(num_regimes)
        self.enabled = bool(enabled)
        self.model_path = Path(model_path).expanduser() if model_path else None
        self._model = None
        self._warned = False
        self.register_buffer("_dummy", torch.zeros(1), persistent=False)

    def _candidate_paths(self) -> list[Path]:
        candidates: list[Path] = []
        if self.model_path is not None:
            candidates.append(self.model_path)

        root = Path(__file__).resolve().parents[3]
        candidates.extend(
            [
                root / "nexus_packaged" / "MMFPS" / "models" / "hmm_regime.pkl",
                root / "MMFPS" / "models" / "hmm_regime.pkl",
                root / "models" / "MMFPS" / "hmm_regime.pkl",
            ]
        )
        return candidates

    def _load_once(self):
        if self._model is not None or not self.enabled:
            return

        for path in self._candidate_paths():
            if not path.exists():
                continue
            try:
                self._model = joblib.load(path)
                return
            except Exception as exc:
                warnings.warn(f"Failed to load HMM model at {path}: {exc}")

        if not self._warned:
            warnings.warn("No HMM regime model found; using uniform regime probabilities.")
            self._warned = True

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        bsz = context.shape[0]
        uniform = torch.full(
            (bsz, self.num_regimes),
            1.0 / float(self.num_regimes),
            device=context.device,
            dtype=context.dtype,
        )

        self._load_once()
        if self._model is None:
            return uniform

        # Per training script contract: context channel-0 is the regime sequence input.
        seq = context[..., 0].detach().cpu().numpy().astype(np.float64)  # (B, T)
        bsz, seq_len = seq.shape
        flat = seq.reshape(-1, 1)
        lengths = [seq_len] * bsz
        try:
            posteriors = self._model.predict_proba(flat, lengths=lengths)
        except Exception as exc:
            if not self._warned:
                warnings.warn(f"HMM inference failed; using uniform regime probabilities. Reason: {exc}")
                self._warned = True
            return uniform

        end_idx = np.cumsum(lengths) - 1
        probs = torch.from_numpy(posteriors[end_idx]).to(device=context.device, dtype=context.dtype)

        if probs.shape[1] > self.num_regimes:
            probs = probs[:, : self.num_regimes]
        elif probs.shape[1] < self.num_regimes:
            pad = torch.zeros(
                (bsz, self.num_regimes - probs.shape[1]),
                device=context.device,
                dtype=context.dtype,
            )
            probs = torch.cat([probs, pad], dim=-1)

        probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return probs


class XGBoostPathScorer(nn.Module):
    """Offline XGBoost scorer for per-path signal.

    If model is unavailable, returns zeros.
    """

    def __init__(self, model_path: Optional[str] = None, enabled: bool = True):
        super().__init__()
        self.enabled = bool(enabled)
        self.model_path = Path(model_path).expanduser() if model_path else None
        self._booster = None
        self._warned = False
        self.register_buffer("_dummy", torch.zeros(1), persistent=False)

    def _candidate_paths(self) -> list[Path]:
        candidates: list[Path] = []
        if self.model_path is not None:
            candidates.append(self.model_path)

        root = Path(__file__).resolve().parents[3]
        candidates.extend(
            [
                root / "nexus_packaged" / "MMFPS" / "models" / "xgb_path_scorer.json",
                root / "MMFPS" / "models" / "xgb_path_scorer.json",
                root / "models" / "MMFPS" / "xgb_path_scorer.json",
            ]
        )
        return candidates

    def _load_once(self):
        if self._booster is not None or not self.enabled:
            return
        if xgb is None:
            if not self._warned:
                warnings.warn("xgboost not installed; using zero XGBoost path scores.")
                self._warned = True
            return

        for path in self._candidate_paths():
            if not path.exists():
                continue
            try:
                booster = xgb.Booster()
                booster.load_model(str(path))
                self._booster = booster
                return
            except Exception as exc:
                warnings.warn(f"Failed to load XGBoost model at {path}: {exc}")

        if not self._warned:
            warnings.warn("No XGBoost model found; using zero XGBoost path scores.")
            self._warned = True

    @staticmethod
    def _build_features(context: torch.Tensor, paths: torch.Tensor, quant_feat: torch.Tensor) -> torch.Tensor:
        """Return per-path features matching train_hmm_xgb.py.

        Feature order:
        - mean return
        - volatility
        - max drawdown
        - trend strength
        - final return
        - range
        """
        del context  # Unused by design for the offline tabular scorer.
        del quant_feat  # Explicitly compute features here to keep train/infer parity.

        prev = paths[..., :-1]
        nxt = paths[..., 1:]
        ret = (nxt - prev) / (prev.abs() + 1e-6)
        mean_return = ret.mean(dim=-1, keepdim=True)
        volatility = ret.std(dim=-1, keepdim=True)

        cumulative = ret.cumsum(dim=-1)
        running_max = cumulative.cummax(dim=-1).values
        drawdown = cumulative - running_max
        max_drawdown = drawdown.min(dim=-1, keepdim=True).values

        trend_strength = paths.diff(dim=-1).abs().mean(dim=-1, keepdim=True)
        final_return = (paths[..., -1:] - paths[..., 0:1]) / (paths[..., 0:1].abs() + 1e-6)
        prange = paths.max(dim=-1, keepdim=True).values - paths.min(dim=-1, keepdim=True).values

        return torch.cat(
            [mean_return, volatility, max_drawdown, trend_strength, final_return, prange],
            dim=-1,
        )

    def forward(
        self,
        context: torch.Tensor,
        paths: torch.Tensor,
        quant_feat: torch.Tensor,
    ) -> torch.Tensor:
        bsz, n_paths, _ = paths.shape
        zeros = torch.zeros((bsz, n_paths, 1), device=context.device, dtype=context.dtype)

        self._load_once()
        if self._booster is None:
            return zeros

        feats = self._build_features(context, paths, quant_feat)
        flat = feats.detach().cpu().numpy().astype(np.float32).reshape(bsz * n_paths, -1)

        try:
            dmat = xgb.DMatrix(flat)
            pred = self._booster.predict(dmat)
        except Exception as exc:
            if not self._warned:
                warnings.warn(f"XGBoost inference failed; using zeros. Reason: {exc}")
                self._warned = True
            return zeros

        pred = np.asarray(pred, dtype=np.float32).reshape(bsz, n_paths, 1)
        return torch.from_numpy(pred).to(device=context.device, dtype=context.dtype)


class FeatureFusion(nn.Module):
    """Fuse context/path/quant/regime/xgb features."""

    def __init__(
        self,
        ctx_dim: int,
        path_dim: int,
        quant_dim: int,
        regime_dim: int,
        xgb_dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.raw_dim = ctx_dim + path_dim + quant_dim + regime_dim + xgb_dim
        self.fusion = nn.Sequential(
            nn.LayerNorm(self.raw_dim),
            nn.Linear(self.raw_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
        )

    def forward(
        self,
        ctx_feat: torch.Tensor,
        path_feat: torch.Tensor,
        quant_feat: torch.Tensor,
        regime_feat: torch.Tensor,
        xgb_score: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, n_paths, _ = path_feat.shape
        ctx_rep = ctx_feat.unsqueeze(1).expand(-1, n_paths, -1)

        combined_raw = torch.cat([ctx_rep, path_feat, quant_feat, regime_feat, xgb_score], dim=-1)
        combined_fused = self.fusion(combined_raw)
        return combined_raw, combined_fused


class BasicScorer(nn.Module):
    """Baseline scorer used for step-6 verification."""

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, combined_fused: torch.Tensor) -> torch.Tensor:
        """combined_fused: (B, 128, D) -> (B, 128)"""
        return self.head(combined_fused).squeeze(-1)


class ResidualMLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = F.gelu(self.fc1(h))
        h = self.fc2(h)
        return x + h


class DiffusionScoreUNet(nn.Module):
    """Stable iterative score-delta predictor for ranking refinement."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_steps: int = 4,
    ):
        super().__init__()
        self.num_steps = int(num_steps)
        self.hidden_dim = int(hidden_dim)

        self.score_proj = nn.Linear(1, hidden_dim)
        self.feature_proj = nn.Linear(feature_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.res1 = ResidualMLP(hidden_dim)
        self.res2 = ResidualMLP(hidden_dim)
        self.gate = nn.Linear(hidden_dim, 1)
        self.out_proj = nn.Linear(hidden_dim, 1)

    @staticmethod
    def _timestep_embedding(step_idx: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        half = max(dim // 2, 1)
        freq = torch.exp(
            -torch.arange(half, device=device, dtype=dtype) * (np.log(10000.0) / max(half - 1, 1))
        )
        t = torch.tensor(float(step_idx), device=device, dtype=dtype)
        angles = t * freq
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=0)
        if emb.shape[0] < dim:
            emb = F.pad(emb, (0, dim - emb.shape[0]))
        return emb[:dim]

    def forward(
        self,
        scores: torch.Tensor,
        combined_raw: torch.Tensor,
        step_idx: int,
    ) -> torch.Tensor:
        """scores: (B, 128), combined_raw: (B, 128, C) -> delta: (B, 128)"""
        bsz, n_paths, _ = combined_raw.shape

        s = self.score_proj(scores.unsqueeze(-1))
        t_emb = self._timestep_embedding(
            step_idx=step_idx,
            dim=self.hidden_dim,
            device=combined_raw.device,
            dtype=combined_raw.dtype,
        ).view(1, 1, -1)
        s = s + t_emb

        kv = self.feature_proj(combined_raw)
        attn_out, _ = self.cross_attn(s, kv, kv)
        h = self.norm1(s + attn_out)

        h = self.res1(h)
        h = self.norm2(h)
        h = self.res2(h)

        delta = self.out_proj(h).squeeze(-1)
        gate = torch.sigmoid(self.gate(h)).squeeze(-1)
        return gate * delta


class SelectorV2(nn.Module):
    """Diffusion-based selector scoring 128 paths conditioned on context."""

    def __init__(
        self,
        feature_dim: int = 144,
        path_len: int = 20,
        num_paths: int = 128,
        d_model: int = 256,
        num_heads: int = 8,
        num_ctx_layers: int = 2,
        use_diffusion: bool = False,
        diffusion_steps: int = 4,
        num_regimes: int = 6,
        use_hmm: bool = True,
        use_xgboost: bool = True,
        hmm_model_path: Optional[str] = None,
        xgb_model_path: Optional[str] = None,
    ):
        super().__init__()

        self.num_paths = int(num_paths)
        self.use_diffusion = bool(use_diffusion)

        self.ctx_encoder = ContextEncoder(
            feature_dim=feature_dim,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_ctx_layers,
        )

        self.path_encoder = PathEncoder(path_len=path_len, d_model=d_model)
        self.quant_features = QuantFeatures(path_len=path_len)

        self.hmm_model = HMMRegimeModel(
            num_regimes=num_regimes,
            model_path=hmm_model_path,
            enabled=use_hmm,
        )

        self.xgb_scorer = XGBoostPathScorer(
            model_path=xgb_model_path,
            enabled=use_xgboost,
        )

        self.fusion = FeatureFusion(
            ctx_dim=d_model,
            path_dim=d_model,
            quant_dim=5,
            regime_dim=num_regimes,
            xgb_dim=1,
            output_dim=d_model,
        )

        self.basic_scorer = BasicScorer(d_model=d_model)

        if self.use_diffusion:
            self.diffusion_unet = DiffusionScoreUNet(
                feature_dim=self.fusion.raw_dim,
                hidden_dim=d_model,
                num_heads=num_heads,
                num_steps=diffusion_steps,
            )

    def forward(self, context: torch.Tensor, paths: torch.Tensor) -> dict:
        """context: (B, 120, F), paths: (B, 128, H)."""
        _, n_paths, _ = paths.shape
        if n_paths != self.num_paths:
            raise ValueError(f"Expected {self.num_paths} paths, got {n_paths}")

        ctx_feat = self.ctx_encoder(context)
        path_feat = self.path_encoder(paths)
        quant_feat = self.quant_features(paths)

        regime_probs = self.hmm_model(context)
        regime_feat = regime_probs.unsqueeze(1).expand(-1, n_paths, -1)

        xgb_score = self.xgb_scorer(context, paths, quant_feat)

        combined_raw, combined_fused = self.fusion(
            ctx_feat=ctx_feat,
            path_feat=path_feat,
            quant_feat=quant_feat,
            regime_feat=regime_feat,
            xgb_score=xgb_score,
        )

        basic_scores = self.basic_scorer(combined_fused)

        if self.use_diffusion:
            refined_scores = basic_scores.detach().clone()
            for t in range(self.diffusion_unet.num_steps):
                delta = self.diffusion_unet(refined_scores, combined_raw, t)
                refined_scores = refined_scores + 0.1 * delta
                # Only downscale when dispersion is too large; do not force unit variance.
                mean = refined_scores.mean(dim=-1, keepdim=True)
                centered = refined_scores - mean
                std = centered.std(dim=-1, keepdim=True).clamp(min=1e-6)
                scale = torch.clamp(std / 3.0, min=1.0)
                refined_scores = mean + centered / scale
                refined_scores = torch.clamp(refined_scores, -5.0, 5.0)
            final_scores = refined_scores
        else:
            final_scores = basic_scores

        weights = F.softmax(final_scores, dim=-1)
        path_ret_series = to_returns(paths)
        path_ret_mean = path_ret_series.mean(dim=-1)

        expected_return = (weights * path_ret_mean).sum(dim=-1)
        prob_up = (weights * (path_ret_mean > 0).float()).sum(dim=-1)

        return {
            "final_scores": final_scores,
            "scores": final_scores,
            "weights": weights,
            "path_returns": path_ret_mean,
            "path_returns_series": path_ret_series,
            "expected_return": expected_return,
            "prob_up": prob_up,
            "regime_probs": regime_probs,
            "xgb_score": xgb_score.squeeze(-1),
            "score_std": final_scores.std(dim=-1),
        }


class SelectorLoss(nn.Module):
    """Hybrid selector objective focused on ranking quality and decision sharpness."""

    def __init__(
        self,
        rank_weight: float = 1.0,
        topk_weight: float = 0.5,
        variance_weight: float = 0.1,
        energy_weight: float = 0.15,
        gap_weight: float = 0.1,
        entropy_weight: float = 0.1,
        soft_weight: float = 0.05,
        rank_temp: float = 0.5,
        margin: float = 0.10,
        pair_margin: float = 0.05,
        pair_margin_weight: float = 0.2,
        topk_margin_alpha: float = 0.5,
        min_std: float = 0.2,
        target_gap: float = 1.0,
        entropy_target: float = 1.2,
        entropy_warmup_epochs: int = 2,
        topk: int = 10,
    ):
        super().__init__()
        self.rank_weight = float(rank_weight)
        self.topk_weight = float(topk_weight)
        self.variance_weight = float(variance_weight)
        self.energy_weight = float(energy_weight)
        self.gap_weight = float(gap_weight)
        self.entropy_weight = float(entropy_weight)
        self.soft_weight = float(soft_weight)
        self.rank_temp = float(rank_temp)
        self.margin = float(margin)
        self.pair_margin = float(pair_margin)
        self.pair_margin_weight = float(pair_margin_weight)
        self.topk_margin_alpha = float(topk_margin_alpha)
        self.min_std = float(min_std)
        self.target_gap = float(target_gap)
        self.entropy_target = float(entropy_target)
        self.entropy_warmup_epochs = int(entropy_warmup_epochs)
        self.current_epoch = 0
        self.topk = int(topk)

    def set_soft_weight(self, weight: float) -> None:
        self.soft_weight = float(weight)

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    @staticmethod
    def _spearman_corr(pred_scores: torch.Tensor, target_scores: torch.Tensor) -> torch.Tensor:
        """Approximate batch Spearman using rank-transformed Pearson correlation."""
        pred_rank = pred_scores.argsort(dim=-1).argsort(dim=-1).float()
        target_rank = target_scores.argsort(dim=-1).argsort(dim=-1).float()
        pred_rank = pred_rank - pred_rank.mean(dim=-1, keepdim=True)
        target_rank = target_rank - target_rank.mean(dim=-1, keepdim=True)
        numer = (pred_rank * target_rank).sum(dim=-1)
        denom = (
            pred_rank.norm(dim=-1) * target_rank.norm(dim=-1)
        ).clamp(min=1e-6)
        return (numer / denom).mean()

    def forward(
        self,
        output: dict,
        paths: torch.Tensor,
        actual_future: torch.Tensor,
    ) -> dict:
        """Build return-distance targets and ranking-focused training loss."""
        path_ret = to_returns(paths)
        real_ret = to_returns(actual_future)

        distance = ((path_ret - real_ret.unsqueeze(1)) ** 2).mean(dim=-1)
        target_scores = -distance

        pred_probs = F.softmax(output["final_scores"], dim=-1)
        target_probs = F.softmax(target_scores, dim=-1)

        loss_soft = ((pred_probs - target_probs) ** 2).mean()

        bsz, n_paths = output["final_scores"].shape
        batch_idx = torch.arange(bsz, device=output["final_scores"].device)
        i = torch.randint(0, n_paths, (bsz,), device=output["final_scores"].device)
        j = torch.randint(0, n_paths, (bsz,), device=output["final_scores"].device)

        s_i = output["final_scores"][batch_idx, i]
        s_j = output["final_scores"][batch_idx, j]
        t_i = target_scores[batch_idx, i]
        t_j = target_scores[batch_idx, j]
        pair_label = (t_i > t_j).float()
        rank_logits = (s_i - s_j) / max(self.rank_temp, 1e-4)
        loss_rank = F.binary_cross_entropy_with_logits(rank_logits, pair_label)
        loss_pair_margin = F.relu(self.pair_margin - (s_i - s_j).abs()).mean()
        loss_rank = loss_rank + self.pair_margin_weight * loss_pair_margin

        topk = min(self.topk, n_paths)
        target_top_idx = torch.topk(target_scores, k=topk, dim=1).indices
        target_top_scores = output["final_scores"].gather(1, target_top_idx)
        target_top_sorted = target_top_scores.sort(dim=-1, descending=True).values

        top1 = target_top_sorted[:, 0]
        topk_last = target_top_sorted[:, -1]
        score_std_per = output["final_scores"].std(dim=-1).detach()
        adaptive_margin = self.margin + self.topk_margin_alpha * score_std_per
        loss_topk_margin = F.relu(adaptive_margin - (top1 - topk_last)).mean()

        pred_top_idx = torch.topk(output["final_scores"], k=topk, dim=1).indices
        pred_top_scores = output["final_scores"].gather(1, pred_top_idx)
        pred_top_sorted = pred_top_scores.sort(dim=-1, descending=True).values
        pred_top_gap = pred_top_sorted[:, 0] - pred_top_sorted[:, -1]
        loss_gap = ((pred_top_gap - self.target_gap) ** 2).mean()

        # Variance floor with smooth hinge; no runaway expansion pressure.
        score_std = output["final_scores"].std(dim=-1).clamp(min=1e-6)
        std_deficit = F.relu(self.min_std - score_std)
        loss_variance = (std_deficit ** 2).mean()

        # Energy contrast with centered scores to prevent scale drift.
        centered_scores = output["final_scores"] - output["final_scores"].mean(dim=-1, keepdim=True)
        energy = -centered_scores
        loss_energy = torch.logsumexp(energy, dim=-1).mean() - energy.mean(dim=-1).mean()

        entropy = -(pred_probs * (pred_probs + 1e-8).log()).sum(dim=-1)
        loss_entropy = ((entropy - self.entropy_target) ** 2).mean()
        entropy_weight_eff = 0.0 if self.current_epoch < self.entropy_warmup_epochs else self.entropy_weight

        loss = (
            self.soft_weight * loss_soft
            + self.rank_weight * loss_rank
            + self.topk_weight * loss_topk_margin
            + self.variance_weight * loss_variance
            + self.energy_weight * loss_energy
            + self.gap_weight * loss_gap
            + entropy_weight_eff * loss_entropy
        )

        with torch.no_grad():
            score_std_mean = output["final_scores"].std(dim=-1).mean()
            target_entropy = -(target_probs * (target_probs + 1e-8).log()).sum(dim=-1).mean()
            non_uniform = (output["final_scores"].std(dim=-1) > 1e-4).float().mean()
            topk_gap = pred_top_gap.mean()
            topk_overlap = (
                (pred_top_idx.unsqueeze(-1) == target_top_idx.unsqueeze(-2)).any(dim=-1).float().mean()
            )
            spearman = self._spearman_corr(output["final_scores"], target_scores)

        return {
            "loss": loss,
            "loss_soft": loss_soft,
            "loss_rank": loss_rank,
            "loss_pair_margin": loss_pair_margin,
            "loss_topk": loss_topk_margin,
            "loss_gap": loss_gap,
            "loss_var": loss_variance,
            "loss_energy": loss_energy,
            "loss_entropy": loss_entropy,
            "score_std": score_std_mean,
            "target_entropy": target_entropy,
            "non_uniform": non_uniform,
            "topk_gap": topk_gap,
            "topk_overlap": topk_overlap,
            "spearman": spearman,
        }
