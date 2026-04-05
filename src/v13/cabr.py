from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from config.project_config import V12_FEATURE_CONSISTENCY_REPORT_PATH, V13_CABR_MODEL_PATH
from src.v12.crowd_emotional_momentum import build_crowd_emotional_momentum
from src.v12.tctl import prepare_tctl_candidates
from src.v12.wfri import REGIME_CLASSES, map_regime_class


GEOMETRY_FEATURE_COLUMNS: tuple[str, ...] = (
    'branch_entropy',
    'path_entropy',
    'path_smoothness',
    'reversal_likelihood',
)

BRANCH_SIGNAL_COLUMNS: tuple[str, ...] = (
    'mean_reversion_likelihood',
    'v10_diversity_score',
    'analog_similarity',
    'leaf_analog_confidence',
    'consensus_strength',
)

CONTEXT_SCALAR_COLUMNS: tuple[str, ...] = (
    'context_regime_confidence',
    'context_atr_percentile_30d',
    'context_rsi_14',
    'context_macd_hist',
    'context_bb_pct',
    'context_days_since_regime_change',
    'context_emotional_momentum',
    'context_emotional_fragility',
    'context_emotional_conviction',
    'context_narrative_age',
)


@dataclass(frozen=True)
class CABRPairBatch:
    branch_a: torch.Tensor
    branch_b: torch.Tensor
    context_a: torch.Tensor
    context_b: torch.Tensor
    label: torch.Tensor


class CABRPairDataset(Dataset[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]):
    def __init__(
        self,
        branch_features: np.ndarray,
        context_features: np.ndarray,
        pairs: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        self.branch_features = np.asarray(branch_features, dtype=np.float32)
        self.context_features = np.asarray(context_features, dtype=np.float32)
        self.pairs = np.asarray(pairs, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.float32)

    def __len__(self) -> int:
        return int(len(self.pairs))

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        left, right = self.pairs[index]
        return (
            self.branch_features[left],
            self.branch_features[right],
            self.context_features[left],
            self.context_features[right],
            np.asarray(self.labels[index], dtype=np.float32),
        )


class BranchEncoder(nn.Module):
    def __init__(self, branch_feature_dim: int, embed_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(branch_feature_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ContextEncoder(nn.Module):
    def __init__(self, context_feature_dim: int, embed_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_feature_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CABR(nn.Module):
    def __init__(
        self,
        branch_feature_dim: int,
        context_feature_dim: int,
        embed_dim: int = 64,
        n_heads: int = 4,
    ):
        super().__init__()
        self.branch_encoder = BranchEncoder(branch_feature_dim, embed_dim)
        self.context_encoder = ContextEncoder(context_feature_dim, embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=n_heads, dropout=0.10, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.score_head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def score(self, branch_features: torch.Tensor, context_features: torch.Tensor) -> torch.Tensor:
        b_emb = self.branch_encoder(branch_features).unsqueeze(1)
        c_emb = self.context_encoder(context_features).unsqueeze(1)
        attn_out, _ = self.cross_attn(query=c_emb, key=b_emb, value=b_emb)
        combined = self.norm(b_emb + attn_out).squeeze(1)
        return self.score_head(combined).squeeze(-1)

    def pairwise_loss(self, score_a: torch.Tensor, score_b: torch.Tensor, label: torch.Tensor, margin: float = 0.05) -> torch.Tensor:
        diff = score_a - score_b
        target = (2.0 * label.float()) - 1.0
        return F.softplus(-target * diff + margin).mean()

    def forward(self, branch_features: torch.Tensor, context_features: torch.Tensor) -> torch.Tensor:
        return self.score(branch_features, context_features)


def _load_pass_features(path: Path = V12_FEATURE_CONSISTENCY_REPORT_PATH) -> tuple[str, ...]:
    payload = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    return tuple(payload.get('legacy_archive_vs_live', {}).get('pass_features', []))


def _rolling_percentile(series: pd.Series, window: int = 512) -> pd.Series:
    def _percentile(window_values: pd.Series) -> float:
        values = np.asarray(window_values, dtype=np.float32)
        if values.size <= 1:
            return 0.5
        rank = np.sum(values <= values[-1]) / float(values.size)
        return float(rank)
    return series.rolling(window=window, min_periods=5).apply(_percentile, raw=False).fillna(0.5)


def _days_since_regime_change(regimes: pd.Series, timestamps: pd.Series) -> pd.Series:
    last_change = None
    previous = None
    values: list[float] = []
    for regime, timestamp in zip(regimes.tolist(), timestamps.tolist(), strict=False):
        ts = pd.Timestamp(timestamp)
        if previous is None or regime != previous:
            last_change = ts
        previous = regime
        delta_days = 0.0 if last_change is None else (ts - last_change).total_seconds() / 86400.0
        values.append(max(delta_days, 0.0))
    return pd.Series(values, index=regimes.index, dtype=np.float32)


def derive_cabr_feature_columns(frame: pd.DataFrame) -> tuple[tuple[str, ...], tuple[str, ...]]:
    passing = tuple(f'bcfe_{name}' for name in _load_pass_features() if f'bcfe_{name}' in frame.columns)
    branch_cols = tuple(
        col
        for col in (passing + GEOMETRY_FEATURE_COLUMNS + BRANCH_SIGNAL_COLUMNS)
        if col in frame.columns
    )
    context_cols = tuple(
        col
        for col in (CONTEXT_SCALAR_COLUMNS + tuple(f'context_regime_{name}' for name in REGIME_CLASSES))
        if col in frame.columns
    )
    return branch_cols, context_cols


def augment_cabr_context(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working['timestamp'] = pd.to_datetime(working['timestamp'], utc=True, errors='coerce')
    working['regime_class'] = working.get('dominant_regime', 'ranging').map(map_regime_class)
    working['context_regime_confidence'] = working.get('hmm_regime_probability', 0.5).astype(float)
    working['context_rsi_14'] = working.get('bcfe_rsi_14', 50.0).astype(float)
    working['context_macd_hist'] = working.get('bcfe_macd_hist', 0.0).astype(float)
    working['context_bb_pct'] = working.get('bcfe_bb_pct', 0.5).astype(float)
    working['context_atr_percentile_30d'] = _rolling_percentile(working.get('bcfe_atr_pct', pd.Series(0.0, index=working.index)).astype(float))
    working['context_days_since_regime_change'] = _days_since_regime_change(working['regime_class'], working['timestamp'])

    cem = build_crowd_emotional_momentum(working)
    if not cem.empty:
        working = working.merge(cem, on='sample_id', how='left')
    working['context_emotional_momentum'] = working.get('cem_momentum', 0.0).fillna(0.0).astype(float)
    working['context_emotional_fragility'] = working.get('cem_fragility', 0.5).fillna(0.5).astype(float)
    working['context_emotional_conviction'] = working.get('cem_conviction', 0.5).fillna(0.5).astype(float)
    working['context_narrative_age'] = working.get('cem_narrative_age', 0).fillna(0).astype(float)

    for regime_name in REGIME_CLASSES:
        working[f'context_regime_{regime_name}'] = (working['regime_class'] == regime_name).astype(float)

    for column in GEOMETRY_FEATURE_COLUMNS + BRANCH_SIGNAL_COLUMNS:
        if column not in working.columns:
            working[column] = 0.0
        working[column] = pd.to_numeric(working[column], errors='coerce').fillna(0.0).astype(float)

    return working


def load_v13_candidate_frames(
    archive: pd.DataFrame,
    *,
    validation_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...], tuple[str, ...]]:
    train_frame, valid_frame, _ = prepare_tctl_candidates(archive, validation_fraction=validation_fraction)
    train_frame = augment_cabr_context(train_frame)
    valid_frame = augment_cabr_context(valid_frame)
    branch_cols, context_cols = derive_cabr_feature_columns(train_frame)
    return train_frame, valid_frame, branch_cols, context_cols


def build_cabr_pairs(
    frame: pd.DataFrame,
    *,
    branch_feature_names: Sequence[str],
    context_feature_names: Sequence[str],
    regime_col: str = 'regime_class',
    hard_negative_ratio: float = 0.33,
    similarity_threshold: float = 0.80,
    max_pairs: int = 60000,
) -> dict[str, Any]:
    working = frame.copy()
    working['timestamp'] = pd.to_datetime(working['timestamp'], utc=True, errors='coerce')
    working = working.sort_values('timestamp').reset_index(drop=True)
    branch_values = working[list(branch_feature_names)].fillna(0.0).to_numpy(dtype=np.float32)
    context_values = working[list(context_feature_names)].fillna(0.0).to_numpy(dtype=np.float32)
    outcomes = working['setl_target_net_unit_pnl'].to_numpy(dtype=np.float32)
    regimes = working.get(regime_col, 'ranging').astype(str).to_numpy()

    pair_rows: list[tuple[int, int]] = []
    labels: list[float] = []
    pair_regimes: list[str] = []
    rng = np.random.default_rng(42)

    for regime in pd.unique(regimes):
        idx = np.where(regimes == regime)[0]
        if idx.size < 4:
            continue
        regime_outcomes = outcomes[idx]
        regime_features = branch_values[idx]

        for left, right in zip(idx[:-1], idx[1:], strict=False):
            if outcomes[left] == outcomes[right]:
                continue
            pair_rows.append((int(left), int(right)))
            labels.append(1.0 if outcomes[left] > outcomes[right] else 0.0)
            pair_regimes.append(str(regime))

        norms = np.linalg.norm(regime_features, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-6)
        cosine = (regime_features @ regime_features.T) / (norms @ norms.T)
        hard_candidates: list[tuple[int, int, float]] = []
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                if cosine[i, j] <= similarity_threshold:
                    continue
                if np.sign(regime_outcomes[i]) == np.sign(regime_outcomes[j]):
                    continue
                left = int(idx[i])
                right = int(idx[j])
                label = 1.0 if outcomes[left] > outcomes[right] else 0.0
                hard_candidates.append((left, right, label))
        if hard_candidates:
            rng.shuffle(hard_candidates)
            keep = max(1, int(len(idx) * hard_negative_ratio))
            for left, right, label in hard_candidates[:keep]:
                pair_rows.append((left, right))
                labels.append(float(label))
                pair_regimes.append(str(regime))

    if len(pair_rows) > max_pairs:
        selection = rng.choice(len(pair_rows), size=max_pairs, replace=False)
        pair_rows = [pair_rows[i] for i in selection.tolist()]
        labels = [labels[i] for i in selection.tolist()]
        pair_regimes = [pair_regimes[i] for i in selection.tolist()]

    return {
        'branch_features': branch_values,
        'context_features': context_values,
        'pairs': np.asarray(pair_rows, dtype=np.int64),
        'labels': np.asarray(labels, dtype=np.float32),
        'pair_regimes': np.asarray(pair_regimes, dtype=object),
        'frame': working,
    }


def _pair_dataloader(pair_payload: dict[str, Any], batch_size: int, shuffle: bool) -> DataLoader:
    dataset = CABRPairDataset(
        pair_payload['branch_features'],
        pair_payload['context_features'],
        pair_payload['pairs'],
        pair_payload['labels'],
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _batch_to_device(batch: tuple, device: torch.device) -> CABRPairBatch:
    branch_a, branch_b, context_a, context_b, label = batch
    return CABRPairBatch(
        branch_a=torch.as_tensor(branch_a, dtype=torch.float32, device=device),
        branch_b=torch.as_tensor(branch_b, dtype=torch.float32, device=device),
        context_a=torch.as_tensor(context_a, dtype=torch.float32, device=device),
        context_b=torch.as_tensor(context_b, dtype=torch.float32, device=device),
        label=torch.as_tensor(label, dtype=torch.float32, device=device),
    )


def evaluate_cabr_pairwise_accuracy(
    model: CABR,
    pair_payload: dict[str, Any],
    *,
    device: str | None = None,
) -> dict[str, Any]:
    if len(pair_payload.get('pairs', [])) == 0:
        return {'overall_accuracy': 0.0, 'per_regime_accuracy': {}}
    target_device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model = model.to(target_device)
    model.eval()
    loader = _pair_dataloader(pair_payload, batch_size=512, shuffle=False)
    correct = 0
    total = 0
    pair_regimes = pair_payload['pair_regimes']
    regime_correct: dict[str, int] = {}
    regime_total: dict[str, int] = {}
    offset = 0
    with torch.no_grad():
        for batch in loader:
            prepared = _batch_to_device(batch, target_device)
            score_a = model.score(prepared.branch_a, prepared.context_a)
            score_b = model.score(prepared.branch_b, prepared.context_b)
            pred = (score_a > score_b).float().detach().cpu().numpy().astype(np.float32)
            label = prepared.label.detach().cpu().numpy().astype(np.float32)
            batch_size = len(label)
            regimes = pair_regimes[offset: offset + batch_size]
            offset += batch_size
            match = pred == label
            correct += int(np.sum(match))
            total += int(batch_size)
            for regime, ok in zip(regimes.tolist(), match.tolist(), strict=False):
                regime_correct[str(regime)] = regime_correct.get(str(regime), 0) + int(bool(ok))
                regime_total[str(regime)] = regime_total.get(str(regime), 0) + 1
    return {
        'overall_accuracy': float(correct / max(total, 1)),
        'per_regime_accuracy': {
            regime: round(float(regime_correct[regime] / max(regime_total[regime], 1)), 6)
            for regime in sorted(regime_total)
        },
    }


def train_cabr_model(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    *,
    branch_feature_names: Sequence[str],
    context_feature_names: Sequence[str],
    device: str | None = None,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 3e-4,
    checkpoint_path: Path = V13_CABR_MODEL_PATH,
) -> dict[str, Any]:
    target_device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    train_pairs = build_cabr_pairs(train_frame, branch_feature_names=branch_feature_names, context_feature_names=context_feature_names)
    valid_pairs = build_cabr_pairs(valid_frame, branch_feature_names=branch_feature_names, context_feature_names=context_feature_names)
    if len(train_pairs['pairs']) == 0:
        raise ValueError('CABR training produced no valid within-regime pairs.')

    model = CABR(len(branch_feature_names), len(context_feature_names)).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(epochs), 1))
    train_loader = _pair_dataloader(train_pairs, batch_size=batch_size, shuffle=True)

    best_acc = -1.0
    best_state: dict[str, Any] | None = None
    loss_history: list[float] = []
    valid_history: list[float] = []

    for _ in range(int(epochs)):
        model.train()
        epoch_loss = 0.0
        steps = 0
        for batch in train_loader:
            prepared = _batch_to_device(batch, target_device)
            score_a = model.score(prepared.branch_a, prepared.context_a)
            score_b = model.score(prepared.branch_b, prepared.context_b)
            loss = model.pairwise_loss(score_a, score_b, prepared.label)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            steps += 1
        scheduler.step()
        loss_history.append(epoch_loss / max(steps, 1))
        evaluation = evaluate_cabr_pairwise_accuracy(model, valid_pairs, device=str(target_device))
        valid_acc = float(evaluation['overall_accuracy'])
        valid_history.append(valid_acc)
        if valid_acc > best_acc:
            best_acc = valid_acc
            best_state = {
                'state_dict': model.state_dict(),
                'branch_feature_names': tuple(branch_feature_names),
                'context_feature_names': tuple(context_feature_names),
                'embed_dim': 64,
                'n_heads': 4,
                'best_accuracy': best_acc,
                'loss_history': loss_history,
                'valid_accuracy_history': valid_history,
            }

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        torch.save(best_state, checkpoint_path)
        model.load_state_dict(best_state['state_dict'])
    return {
        'model': model,
        'train_pairs': train_pairs,
        'valid_pairs': valid_pairs,
        'best_accuracy': float(best_acc),
        'loss_history': [float(v) for v in loss_history],
        'valid_accuracy_history': [float(v) for v in valid_history],
        'device': str(target_device),
    }


def load_cabr_model(path: Path = V13_CABR_MODEL_PATH, *, map_location: str | None = None) -> tuple[CABR, tuple[str, ...], tuple[str, ...], dict[str, Any]]:
    payload = torch.load(path, map_location=map_location or 'cpu')
    branch_feature_names = tuple(payload['branch_feature_names'])
    context_feature_names = tuple(payload['context_feature_names'])
    model = CABR(
        branch_feature_dim=len(branch_feature_names),
        context_feature_dim=len(context_feature_names),
        embed_dim=int(payload.get('embed_dim', 64)),
        n_heads=int(payload.get('n_heads', 4)),
    )
    model.load_state_dict(payload['state_dict'])
    model.eval()
    return model, branch_feature_names, context_feature_names, payload


def score_cabr_model(
    model: CABR,
    frame: pd.DataFrame,
    *,
    branch_feature_names: Sequence[str],
    context_feature_names: Sequence[str],
    device: str | None = None,
) -> np.ndarray:
    target_device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model = model.to(target_device)
    model.eval()
    branch_values = np.asarray(frame[list(branch_feature_names)].fillna(0.0).to_numpy(dtype=np.float32), order='C').copy()
    context_values = np.asarray(frame[list(context_feature_names)].fillna(0.0).to_numpy(dtype=np.float32), order='C').copy()
    with torch.no_grad():
        branch_tensor = torch.from_numpy(branch_values).to(device=target_device)
        context_tensor = torch.from_numpy(context_values).to(device=target_device)
        scores = model.score(branch_tensor, context_tensor).detach().cpu().numpy().astype(np.float32)
    return scores
