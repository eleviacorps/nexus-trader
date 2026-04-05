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

from config.project_config import V12_FEATURE_CONSISTENCY_REPORT_PATH, V12_TCTL_MODEL_PATH
from src.backtest.engine import capital_backtest_from_unit_pnl
from src.v11 import SETL_FEATURES
from src.v11.path_conditioned_outcome import apply_pcop_model, reweight_branches, train_pcop_model
from src.v11.research_backtest import augment_v11_context, build_stage_dataset, score_selector_model, train_selector_model
from src.v12.wfri import map_regime_class
from src.v12.bar_consistent_features import (
    PRICE_FEATURE_COLUMNS,
    compute_bar_consistent_features,
    compute_online_feature_frame,
    load_default_raw_bars,
)


TCTL_BASE_FEATURES: tuple[str, ...] = tuple(SETL_FEATURES) + (
    "v10_temperature",
    "v10_minority_rescue",
)


@dataclass(frozen=True)
class TCTLThreshold:
    threshold: float
    participation_rate: float
    avg_unit_pnl: float


class PairIndexDataset(Dataset[tuple[int, int]]):
    def __init__(self, pairs: np.ndarray) -> None:
        self.pairs = np.asarray(pairs, dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self.pairs))

    def __getitem__(self, index: int) -> tuple[int, int]:
        item = self.pairs[index]
        return int(item[0]), int(item[1])


class TCTLRanker(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, max(hidden_dim // 2, 16)),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(max(hidden_dim // 2, 16), 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.score_net(features).squeeze(-1)

    @staticmethod
    def pairwise_loss(score_better: torch.Tensor, score_worse: torch.Tensor, margin: float = 0.1) -> torch.Tensor:
        return torch.mean(F.softplus(-(score_better - score_worse - margin)))


def _time_split(frame: pd.DataFrame, validation_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_ids = np.asarray(sorted(frame["sample_id"].unique().tolist()), dtype=np.int64)
    validation_count = min(max(int(len(sample_ids) * validation_fraction), 1), max(len(sample_ids) - 1, 1))
    split_at = len(sample_ids) - validation_count
    train_ids = set(sample_ids[:split_at].tolist())
    valid_ids = set(sample_ids[split_at:].tolist())
    return (
        frame.loc[frame["sample_id"].isin(train_ids)].copy().reset_index(drop=True),
        frame.loc[frame["sample_id"].isin(valid_ids)].copy().reset_index(drop=True),
    )


def load_bcfe_pass_features(path: Path = V12_FEATURE_CONSISTENCY_REPORT_PATH) -> tuple[str, ...]:
    if not path.exists():
        return tuple(f"bcfe_{name}" for name in PRICE_FEATURE_COLUMNS)
    payload = json.loads(path.read_text(encoding="utf-8"))
    passed = payload.get("legacy_archive_vs_live", {}).get("pass_features", [])
    return tuple(f"bcfe_{name}" for name in passed if name in PRICE_FEATURE_COLUMNS)


def _fit_v11_stage_scores(train_frame: pd.DataFrame, valid_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_frame = train_frame.copy()
    valid_frame = valid_frame.copy()

    selector_model = None
    selector_features: tuple[str, ...] = ()
    try:
        selector_model, selector_features = train_selector_model(train_frame)
    except Exception:
        selector_model, selector_features = None, ()
    if selector_model is not None and selector_features:
        train_frame["selector_score"] = score_selector_model(selector_model, train_frame, feature_names=selector_features)
        valid_frame["selector_score"] = score_selector_model(selector_model, valid_frame, feature_names=selector_features)
    else:
        train_frame["selector_score"] = train_frame.get("composite_score", 0.5).to_numpy(dtype=np.float32)
        valid_frame["selector_score"] = valid_frame.get("composite_score", 0.5).to_numpy(dtype=np.float32)

    pcop5_model, pcop5_features = None, ()
    try:
        pcop5_model, pcop5_features = train_pcop_model(train_frame.assign(selector_score=train_frame["selector_score"]), stage_bars=5)
    except Exception:
        pcop5_model, pcop5_features = None, ()
    train_frame["pcop_survival_score_5m"] = apply_pcop_model(
        pcop5_model,
        train_frame.assign(selector_score=train_frame["selector_score"]),
        stage_bars=5,
        feature_names=pcop5_features,
    )
    valid_frame["pcop_survival_score_5m"] = apply_pcop_model(
        pcop5_model,
        valid_frame.assign(selector_score=valid_frame["selector_score"]),
        stage_bars=5,
        feature_names=pcop5_features,
    )
    train_frame = reweight_branches(train_frame, survival_scores=train_frame["pcop_survival_score_5m"].to_numpy(dtype=np.float32)).rename(columns={"pcop_conditioned_score": "pcop_conditioned_score_5m"})
    valid_frame = reweight_branches(valid_frame, survival_scores=valid_frame["pcop_survival_score_5m"].to_numpy(dtype=np.float32)).rename(columns={"pcop_conditioned_score": "pcop_conditioned_score_5m"})

    pcop10_model, pcop10_features = None, ()
    try:
        pcop10_model, pcop10_features = train_pcop_model(train_frame.assign(selector_score=train_frame["selector_score"]), stage_bars=10)
    except Exception:
        pcop10_model, pcop10_features = None, ()
    train_frame["pcop_survival_score_10m"] = apply_pcop_model(
        pcop10_model,
        train_frame.assign(selector_score=train_frame["selector_score"]),
        stage_bars=10,
        feature_names=pcop10_features,
    )
    valid_frame["pcop_survival_score_10m"] = apply_pcop_model(
        pcop10_model,
        valid_frame.assign(selector_score=valid_frame["selector_score"]),
        stage_bars=10,
        feature_names=pcop10_features,
    )
    train_frame["pcop_conditioned_score_10m"] = (
        0.55 * train_frame["selector_score"].to_numpy(dtype=np.float32)
        + 0.45 * train_frame["pcop_survival_score_10m"].to_numpy(dtype=np.float32)
    )
    valid_frame["pcop_conditioned_score_10m"] = (
        0.55 * valid_frame["selector_score"].to_numpy(dtype=np.float32)
        + 0.45 * valid_frame["pcop_survival_score_10m"].to_numpy(dtype=np.float32)
    )
    return train_frame, valid_frame


def _attach_bcfe_features(
    frame: pd.DataFrame,
    raw_feature_frame: pd.DataFrame,
    *,
    pass_feature_names: Sequence[str],
) -> pd.DataFrame:
    working = frame.copy()
    timestamps = pd.to_datetime(working["timestamp"], utc=True, errors="coerce") + pd.to_timedelta(working["stage_bars"], unit="m")
    aligned = raw_feature_frame.reindex(timestamps, method="pad")
    aligned.index = working.index
    for feature in pass_feature_names:
        source = feature.replace("bcfe_", "", 1)
        working[feature] = aligned[source].to_numpy(dtype=np.float32) if source in aligned.columns else 0.0
    return working


def _load_bcfe_source_bars(
    frame: pd.DataFrame,
    *,
    raw_bars: pd.DataFrame | None,
    warmup_bars: int,
) -> pd.DataFrame:
    if raw_bars is not None:
        return raw_bars.copy()
    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if timestamps.isna().all():
        raise ValueError("TCTL candidate frame has no valid timestamps for BCFE alignment.")
    start = timestamps.min() - pd.Timedelta(minutes=max(int(warmup_bars), 1) + 30)
    end = timestamps.max() + pd.Timedelta(minutes=16)
    return load_default_raw_bars(start=start, end=end)


def build_bcfe_feature_frame(
    frame: pd.DataFrame,
    *,
    raw_bars: pd.DataFrame | None = None,
    replay_mode: str = "archive",
    warmup_bars: int = 200,
) -> pd.DataFrame:
    source_bars = _load_bcfe_source_bars(frame, raw_bars=raw_bars, warmup_bars=warmup_bars)
    if str(replay_mode).strip().lower() == "online":
        return compute_online_feature_frame(source_bars, warmup_bars=warmup_bars)
    return compute_bar_consistent_features(source_bars)


def replay_candidates_with_bcfe_mode(
    frame: pd.DataFrame,
    *,
    pass_feature_names: Sequence[str] | None = None,
    raw_bars: pd.DataFrame | None = None,
    replay_mode: str = "online",
    warmup_bars: int = 200,
) -> pd.DataFrame:
    bcfe_names = tuple(pass_feature_names or load_bcfe_pass_features())
    raw_feature_frame = build_bcfe_feature_frame(
        frame,
        raw_bars=raw_bars,
        replay_mode=replay_mode,
        warmup_bars=warmup_bars,
    )
    return _attach_bcfe_features(frame, raw_feature_frame, pass_feature_names=bcfe_names)


def replay_candidates_with_online_bcfe(
    frame: pd.DataFrame,
    *,
    pass_feature_names: Sequence[str] | None = None,
    raw_bars: pd.DataFrame | None = None,
    warmup_bars: int = 200,
) -> pd.DataFrame:
    return replay_candidates_with_bcfe_mode(
        frame,
        pass_feature_names=pass_feature_names,
        raw_bars=raw_bars,
        replay_mode="online",
        warmup_bars=warmup_bars,
    )


def _stage_frame(
    frame: pd.DataFrame,
    *,
    stage_bars: int,
    selector_score_column: str,
    pcop_score_column: str | None,
    stage_name: str,
) -> pd.DataFrame:
    stage = build_stage_dataset(frame, stage_bars=stage_bars, selector_score_column=selector_score_column, pcop_score_column=pcop_score_column).copy()
    stage["stage_name"] = stage_name
    stage["stage_bars"] = int(stage_bars)
    return stage


def prepare_tctl_candidates(
    frame: pd.DataFrame,
    *,
    validation_fraction: float = 0.2,
    pass_feature_names: Sequence[str] | None = None,
    raw_bars: pd.DataFrame | None = None,
    warmup_bars: int = 200,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    working = augment_v11_context(frame)
    train_frame, valid_frame = _time_split(working, validation_fraction=validation_fraction)
    train_frame, valid_frame = _fit_v11_stage_scores(train_frame, valid_frame)

    train_candidates = pd.concat(
        [
            _stage_frame(train_frame, stage_bars=0, selector_score_column="selector_score", pcop_score_column=None, stage_name="open"),
            _stage_frame(train_frame, stage_bars=5, selector_score_column="selector_score", pcop_score_column="pcop_conditioned_score_5m", stage_name="pcop_5m"),
            _stage_frame(train_frame, stage_bars=10, selector_score_column="selector_score", pcop_score_column="pcop_conditioned_score_10m", stage_name="pcop_10m"),
        ],
        ignore_index=True,
    )
    valid_candidates = pd.concat(
        [
            _stage_frame(valid_frame, stage_bars=0, selector_score_column="selector_score", pcop_score_column=None, stage_name="open"),
            _stage_frame(valid_frame, stage_bars=5, selector_score_column="selector_score", pcop_score_column="pcop_conditioned_score_5m", stage_name="pcop_5m"),
            _stage_frame(valid_frame, stage_bars=10, selector_score_column="selector_score", pcop_score_column="pcop_conditioned_score_10m", stage_name="pcop_10m"),
        ],
        ignore_index=True,
    )

    bcfe_names = tuple(pass_feature_names or load_bcfe_pass_features())
    raw_feature_frame = build_bcfe_feature_frame(
        working,
        raw_bars=raw_bars,
        replay_mode="archive",
        warmup_bars=warmup_bars,
    )
    train_candidates = _attach_bcfe_features(train_candidates, raw_feature_frame, pass_feature_names=bcfe_names)
    valid_candidates = _attach_bcfe_features(valid_candidates, raw_feature_frame, pass_feature_names=bcfe_names)

    feature_names = tuple(
        name
        for name in (TCTL_BASE_FEATURES + tuple(bcfe_names))
        if name in train_candidates.columns and name in valid_candidates.columns
    )
    return train_candidates, valid_candidates, feature_names


def build_training_pairs(
    frame: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    regime_col: str = "dominant_regime",
    window_days: int = 30,
    max_pairs_per_window: int = 4096,
    similarity_threshold: float = 0.85,
) -> np.ndarray:
    working = frame.copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True, errors="coerce")
    if regime_col in working.columns:
        working["_pair_regime"] = working[regime_col].map(map_regime_class)
    else:
        working["_pair_regime"] = "ranging"
    working = working.sort_values("timestamp").reset_index(drop=True)
    pairs: list[tuple[int, int]] = []
    if working.empty:
        return np.zeros((0, 2), dtype=np.int64)

    start = working["timestamp"].min()
    end = working["timestamp"].max()
    current = start
    rng = np.random.default_rng(42)
    while current <= end:
        window_end = current + pd.Timedelta(days=window_days)
        window = working.loc[(working["timestamp"] >= current) & (working["timestamp"] < window_end)].copy()
        if len(window) >= 4:
            for _, regime_window in window.groupby("_pair_regime", sort=False):
                if len(regime_window) < 4:
                    continue
                outcomes = regime_window["setl_target_net_unit_pnl"].to_numpy(dtype=np.float32)
                order = np.argsort(outcomes)
                better = regime_window.index.to_numpy(dtype=np.int64)[order[::-1]]
                worse = regime_window.index.to_numpy(dtype=np.int64)[order]
                for left in better[: min(len(better), 32)]:
                    worse_pool = [
                        right
                        for right in worse[: min(len(worse), 32)]
                        if left != right and working.loc[left, "setl_target_net_unit_pnl"] > working.loc[right, "setl_target_net_unit_pnl"]
                    ]
                    for right in worse_pool[:3]:
                        pairs.append((int(left), int(right)))

                values = regime_window[list(feature_names)].fillna(0.0).to_numpy(dtype=np.float32)
                norms = np.linalg.norm(values, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-6)
                cosine = (values @ values.T) / (norms @ norms.T)
                signs = np.sign(outcomes)
                candidate_pairs = np.argwhere((cosine >= similarity_threshold) & (cosine < 0.9999))
                hard_negatives: list[tuple[int, int]] = []
                for left_pos, right_pos in candidate_pairs.tolist():
                    if left_pos >= right_pos:
                        continue
                    if signs[left_pos] == signs[right_pos]:
                        continue
                    better_idx = int(regime_window.index[left_pos] if outcomes[left_pos] > outcomes[right_pos] else regime_window.index[right_pos])
                    worse_idx = int(regime_window.index[right_pos] if better_idx == regime_window.index[left_pos] else regime_window.index[left_pos])
                    hard_negatives.append((better_idx, worse_idx))
                if hard_negatives:
                    rng.shuffle(hard_negatives)
                    pairs.extend(hard_negatives[: max(1, min(len(hard_negatives), len(regime_window) // 2))])
        current = window_end

    if len(pairs) > max_pairs_per_window * 12:
        rng.shuffle(pairs)
        pairs = pairs[: max_pairs_per_window * 12]
    return np.asarray(pairs, dtype=np.int64)


def _feature_tensor(frame: pd.DataFrame, feature_names: Sequence[str], device: torch.device) -> torch.Tensor:
    values = np.asarray(frame[list(feature_names)].fillna(0.0).to_numpy(dtype=np.float32), order="C").copy()
    return torch.from_numpy(values).to(device=device)


def train_tctl_model(
    frame: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    device: str | None = None,
    epochs: int = 12,
    batch_size: int = 512,
    lr: float = 1e-3,
    checkpoint_path: Path = V12_TCTL_MODEL_PATH,
) -> dict[str, Any]:
    if not feature_names:
        raise ValueError("TCTL requires at least one feature.")
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    pairs = build_training_pairs(frame, feature_names=feature_names)
    if len(pairs) == 0:
        raise ValueError("TCTL training produced no valid ranking pairs.")

    model = TCTLRanker(feature_dim=len(feature_names)).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    dataset = PairIndexDataset(pairs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    features = _feature_tensor(frame, feature_names, target_device)
    losses: list[float] = []

    model.train()
    for _ in range(int(epochs)):
        epoch_loss = 0.0
        step_count = 0
        for better_idx, worse_idx in loader:
            optimizer.zero_grad(set_to_none=True)
            better_scores = model(features[better_idx])
            worse_scores = model(features[worse_idx])
            loss = model.pairwise_loss(better_scores, worse_scores)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            step_count += 1
        losses.append(epoch_loss / max(step_count, 1))

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_names": tuple(feature_names),
            "hidden_dim": 128,
            "loss_history": losses,
        },
        checkpoint_path,
    )
    return {
        "model": model,
        "feature_names": tuple(feature_names),
        "pair_count": int(len(pairs)),
        "loss_history": losses,
        "device": str(target_device),
    }


def load_tctl_model(path: Path = V12_TCTL_MODEL_PATH, *, map_location: str | None = None) -> tuple[TCTLRanker, tuple[str, ...]]:
    payload = torch.load(path, map_location=map_location or "cpu")
    feature_names = tuple(payload["feature_names"])
    model = TCTLRanker(feature_dim=len(feature_names), hidden_dim=int(payload.get("hidden_dim", 128)))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, feature_names


def score_tctl_model(model: TCTLRanker, frame: pd.DataFrame, *, feature_names: Sequence[str], device: str | None = None) -> np.ndarray:
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(target_device)
    model.eval()
    with torch.no_grad():
        features = _feature_tensor(frame, feature_names, target_device)
        scores = model(features).detach().cpu().numpy().astype(np.float32)
    return scores


def evaluate_pairwise_accuracy(model: TCTLRanker, frame: pd.DataFrame, pairs: np.ndarray, *, feature_names: Sequence[str], device: str | None = None) -> float:
    if len(pairs) == 0:
        return 0.0
    scores = score_tctl_model(model, frame, feature_names=feature_names, device=device)
    wins = np.sum(scores[pairs[:, 0]] > scores[pairs[:, 1]])
    return float(wins / max(len(pairs), 1))


def optimize_tctl_threshold(scores: np.ndarray, pnl: np.ndarray) -> TCTLThreshold:
    predicted = np.asarray(scores, dtype=np.float32)
    actual = np.asarray(pnl, dtype=np.float32)
    best = TCTLThreshold(threshold=float(np.min(predicted)), participation_rate=1.0, avg_unit_pnl=float(np.mean(actual)) if len(actual) else 0.0)
    if predicted.size == 0:
        return best
    candidates = sorted(set(np.quantile(predicted, [0.35, 0.45, 0.55, 0.65, 0.75, 0.85]).astype(np.float32).tolist() + [float(np.min(predicted))]))
    for threshold in candidates:
        active = predicted >= float(threshold)
        participation = float(np.mean(active))
        if participation < 0.15 or participation > 0.45:
            continue
        pnl_active = float(np.mean(actual[active])) if np.any(active) else 0.0
        score = pnl_active * np.sqrt(max(participation, 1e-6))
        incumbent = best.avg_unit_pnl * np.sqrt(max(best.participation_rate, 1e-6))
        if score > incumbent:
            best = TCTLThreshold(threshold=float(threshold), participation_rate=participation, avg_unit_pnl=pnl_active)
    if best.participation_rate > 0.45:
        fallback = float(np.quantile(predicted, 0.70))
        active = predicted >= fallback
        best = TCTLThreshold(
            threshold=fallback,
            participation_rate=float(np.mean(active)),
            avg_unit_pnl=float(np.mean(actual[active])) if np.any(active) else 0.0,
        )
    return best


def evaluate_stage_policy(
    frame: pd.DataFrame,
    *,
    score_column: str,
    threshold: float,
) -> dict[str, Any]:
    ranked = frame.sort_values(["sample_id", score_column], ascending=[True, False], kind="mergesort").groupby("sample_id", sort=False).head(1).copy()
    scores = ranked[score_column].to_numpy(dtype=np.float32)
    pnl = ranked["setl_target_net_unit_pnl"].to_numpy(dtype=np.float32)
    active = scores >= float(threshold)
    chosen = np.where(active, pnl, 0.0).astype(np.float32)
    wins = int(np.sum((chosen > 0.0) & active))
    trade_count = int(np.sum(active))
    capital = capital_backtest_from_unit_pnl(chosen, initial_capital=1000.0, risk_fraction=0.02)
    stage_usage = ranked.loc[active, "stage_name"].value_counts().to_dict()
    return {
        "trade_count": trade_count,
        "hold_count": int(len(active) - trade_count),
        "participation": round(float(np.mean(active)) if len(active) else 0.0, 6),
        "win_rate": round(float(wins / max(trade_count, 1)), 6),
        "avg_unit_pnl": round(float(np.mean(pnl[active])) if trade_count else 0.0, 6),
        "cumulative_unit_pnl": round(float(np.sum(chosen)), 6),
        "stage_usage": {name: int(count) for name, count in stage_usage.items()},
        "capital_backtest_usd_1000": capital,
    }
