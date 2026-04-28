"""Generate dataset for diffusion-based residual adjuster."""

from __future__ import annotations

import argparse
from math import dist
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_selector_data_dir(user_path: str | None) -> Path:
    if user_path:
        p = Path(user_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Selector dataset path not found: {p}")
        return p

    root = _repo_root()
    preferred = root / "nexus_packaged" / "MMFPS" / "selector" / "dataset"
    fallback = root / "nexus_packaged" / "outputs" / "MMFPS" / "selector_data"
    if preferred.exists():
        return preferred
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"No selector dataset found at {preferred} or {fallback}")


def _resolve_models_dir(user_path: str | None) -> Path:
    if user_path:
        p = Path(user_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Models directory not found: {p}")
        return p
    return _repo_root() / "nexus_packaged" / "MMFPS" / "models"


def _chunk_paths(data_dir: Path, max_chunks: int | None) -> list[Path]:
    chunks = sorted(data_dir.glob("chunk_*.npz"))
    if not chunks:
        chunks = sorted(data_dir.glob("*.npz"))
    if not chunks:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")
    if max_chunks is not None:
        chunks = chunks[:max_chunks]
    return chunks


def _max_drawdown_from_returns(ret: np.ndarray) -> np.ndarray:
    cumulative = np.cumsum(ret, axis=-1)
    running_max = np.maximum.accumulate(cumulative, axis=-1)
    drawdown = cumulative - running_max
    return drawdown.min(axis=-1)


def _build_path_features(paths_2d: np.ndarray) -> np.ndarray:
    """paths_2d: (M, H) -> features: (M, 6)."""
    prev = paths_2d[:, :-1]
    nxt = paths_2d[:, 1:]
    ret = (nxt - prev) / (np.abs(prev) + 1e-6)
    mean_return = ret.mean(axis=1)
    volatility = ret.std(axis=1)
    max_drawdown = _max_drawdown_from_returns(ret)
    trend_strength = np.mean(np.abs(np.diff(paths_2d, axis=1)), axis=1)
    final_return = (paths_2d[:, -1] - paths_2d[:, 0]) / (np.abs(paths_2d[:, 0]) + 1e-6)
    prange = paths_2d.max(axis=1) - paths_2d.min(axis=1)
    return np.stack(
        [mean_return, volatility, max_drawdown, trend_strength, final_return, prange],
        axis=1,
    ).astype(np.float32)


def _extract_scale(series: np.ndarray, length: int) -> np.ndarray:
    """series: (B, T) -> (B, length), edge-padded on the left if needed."""
    bsz, cur_len = series.shape
    if cur_len >= length:
        return series[:, -length:].astype(np.float32)
    pad_count = length - cur_len
    left = np.repeat(series[:, :1], pad_count, axis=1)
    return np.concatenate([left, series], axis=1).astype(np.float32)


def _build_schedule(timesteps: int, beta_start: float, beta_end: float) -> np.ndarray:
    betas = np.linspace(beta_start, beta_end, timesteps, dtype=np.float32)
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas, axis=0)
    return np.concatenate([np.array([1.0], dtype=np.float32), alpha_bars], axis=0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate adjuster diffusion dataset from selector chunks")
    parser.add_argument("--selector-data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--models-dir", type=str, default=None)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=2e-2)
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale-factor", type=float, default=2.0)
    parser.add_argument("--augment-factor", type=int, default=1)
    parser.add_argument("--resample-per-chunk", type=int, default=1)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    selector_data_dir = _resolve_selector_data_dir(args.selector_data_dir)
    root = _repo_root()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else root / "nexus_packaged" / "MMFPS" / "adjuster" / "dataset"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    models_dir = _resolve_models_dir(args.models_dir)
    hmm_path = models_dir / "hmm_regime.pkl"
    xgb_path = models_dir / "xgb_path_scorer.json"
    if not hmm_path.exists():
        raise FileNotFoundError(f"HMM model missing: {hmm_path}")
    if not xgb_path.exists():
        raise FileNotFoundError(f"XGB model missing: {xgb_path}")

    hmm = joblib.load(hmm_path)
    booster = xgb.Booster()
    booster.load_model(str(xgb_path))

    max_chunks = args.max_chunks if args.max_chunks > 0 else None
    chunks = _chunk_paths(selector_data_dir, max_chunks=max_chunks)
    alpha_bars = _build_schedule(args.timesteps, args.beta_start, args.beta_end)

    print("=== Generating Adjuster Dataset ===")
    print(f"selector_data_dir={selector_data_dir}")
    print(f"output_dir={output_dir}")
    print(f"hmm={hmm_path}")
    print(f"xgb={xgb_path}")
    print(f"timesteps={args.timesteps}")
    print(f"chunks={len(chunks)}")
    print(f"[AdjusterDataset] Using fixed scale_factor={args.scale_factor}")

    total_samples = 0
    sample_cap = args.max_samples if args.max_samples > 0 else None
    out_idx = 1

    for chunk_path in chunks:
        arr = np.load(chunk_path)
        contexts = arr["contexts"].astype(np.float32)   # (B, T, C)
        paths = arr["paths"].astype(np.float32)         # (B, N, H)
        futures = arr["futures"].astype(np.float32)     # (B, H)

        bsz = contexts.shape[0]
        if sample_cap is not None and total_samples >= sample_cap:
            break
        effective_bsz = bsz

        if sample_cap is not None and total_samples + effective_bsz > sample_cap:
            keep = sample_cap - total_samples
            selected = selected[:keep]
            futures = futures[:keep]
            contexts = contexts[:keep]
            clean_delta = clean_delta[:keep]
            noisy_delta = noisy_delta[:keep]
            t = t[:keep]
            bsz = keep

        all_selected = []
        all_futures = []
        all_contexts = []

        for _ in range(args.resample_per_chunk):

            dist = ((paths - futures[:, None, :]) ** 2).mean(axis=-1)
            top_k = 4
            sorted_idx = np.argsort(dist, axis=1)

            idx = np.zeros(bsz, dtype=np.int32)

            for i in range(bsz):
                candidates = sorted_idx[i, :top_k]
                weights = np.array([0.5, 0.3, 0.15, 0.05])
                weights = weights / weights.sum()
                idx[i] = np.random.choice(candidates, p=weights)

            selected = paths[np.arange(bsz), idx]

            # small noise
            noise_sel = np.random.normal(0, 0.005, size=selected.shape)
            selected = selected + noise_sel

            all_selected.append(selected)
            all_futures.append(futures)
            all_contexts.append(contexts)

        # merge
        selected = np.concatenate(all_selected, axis=0)
        futures = np.concatenate(all_futures, axis=0)
        contexts = np.concatenate(all_contexts, axis=0)
        bsz = selected.shape[0]

        # 2) Residual target with amplified target magnitude signal.
        target_future = futures
        scale = args.scale_factor  # already passed
        clean_future = futures  # (B, H)

        if args.augment_factor > 1:
            sel_list = []
            fut_list = []
            ctx_list = []

            for _ in range(args.augment_factor):
                noise = np.random.normal(0, 0.003, size=selected.shape)
                sel_list.append(selected + noise)
                fut_list.append(futures)
                ctx_list.append(contexts)

            selected = np.concatenate(sel_list, axis=0)
            futures = np.concatenate(fut_list, axis=0)
            contexts = np.concatenate(ctx_list, axis=0)
            bsz = selected.shape[0]

            # recompute delta after augmentation
            clean_future = (futures - selected) * scale

        # 3) Diffusion noising.
        t = rng.integers(1, args.timesteps + 1, size=(bsz,), dtype=np.int32)
        noise = rng.standard_normal(clean_future.shape, dtype=np.float32)
        ab = alpha_bars[t][:, None]
        noisy_future = np.sqrt(ab) * clean_future + np.sqrt(1.0 - ab) * noise 

        # 4) Multi-scale context from channel-0.
        ctx_price = contexts[:, :, 0]  # (B, T)
        ctx_120 = _extract_scale(ctx_price, 120)
        ctx_240 = _extract_scale(ctx_price, 240)
        ctx_480 = _extract_scale(ctx_price, 480)

        # 5) HMM regime probs from ctx_120 sequence.
        seq = ctx_120.reshape(-1, 1).astype(np.float64)
        lengths = [ctx_120.shape[1]] * bsz
        post = hmm.predict_proba(seq, lengths=lengths).reshape(bsz, ctx_120.shape[1], -1)
        regime = post.mean(axis=1).astype(np.float32)  # (B, R)

        # 6) Quant + XGB score for selected path.
        selected_features = _build_path_features(selected)  # (B, 6)
        dmat = xgb.DMatrix(selected_features)
        xgb_score = booster.predict(dmat).reshape(-1, 1).astype(np.float32)

        prev = selected[:, :-1]
        nxt = selected[:, 1:]
        sel_ret = (nxt - prev) / (np.abs(prev) + 1e-6)
        sel_std = sel_ret.std(axis=1)
        sel_mean = sel_ret.mean(axis=1)
        sel_mdd = _max_drawdown_from_returns(sel_ret)
        sel_range = selected.max(axis=1) - selected.min(axis=1)
        quant = np.stack([sel_std, sel_mean, sel_mdd, sel_range], axis=1).astype(np.float32)

        out_file = output_dir / f"chunk_{out_idx:04d}.npz"
        np.savez_compressed(
            out_file,
            noisy_future=noisy_future.astype(np.float32),
            clean_future=clean_future.astype(np.float32),
            t=t.astype(np.int32),
            selected_path=selected.astype(np.float32),
            ctx_120=ctx_120.astype(np.float32),
            ctx_240=ctx_240.astype(np.float32),
            ctx_480=ctx_480.astype(np.float32),
            regime=regime.astype(np.float32),
            quant=quant.astype(np.float32),
            xgb=xgb_score.astype(np.float32),
        )

        total_samples += bsz
        print(f"saved {out_file.name}: samples={bsz}, total={total_samples}")
        out_idx += 1

        if sample_cap is not None and total_samples >= sample_cap:
            break

    meta_file = output_dir / "meta.npz"
    np.savez_compressed(
        meta_file,
        timesteps=np.array([args.timesteps], dtype=np.int32),
        beta_start=np.array([args.beta_start], dtype=np.float32),
        beta_end=np.array([args.beta_end], dtype=np.float32),
        total_samples=np.array([total_samples], dtype=np.int32),
    )
    print(f"saved meta: {meta_file}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
