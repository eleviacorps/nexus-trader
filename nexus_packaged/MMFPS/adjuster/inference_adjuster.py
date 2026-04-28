"""Inference for diffusion-based adjuster."""

from __future__ import annotations
import xgboost as xgb
import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import scale
import torch

_p = Path(__file__).resolve().parents[2]
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))

from MMFPS.adjuster.adjuster_diffusion_model import (  # noqa: E402
    AdjusterDiffusionModel,
    DiffusionConfig,
    DiffusionSchedule,
)
from MMFPS.adjuster.metrics import evaluate_adjustment, print_metrics  # noqa: E402


def _load_sample(data_dir: Path, sample_index: int) -> dict[str, np.ndarray]:
    chunks = sorted(data_dir.glob("chunk_*.npz"))
    if not chunks:
        chunks = sorted(data_dir.glob("*.npz"))
    chunks = [c for c in chunks if c.name != "meta.npz"]
    if not chunks:
        raise FileNotFoundError(f"No dataset chunks in {data_dir}")

    rem = sample_index
    for c in chunks:
        arr = np.load(c)
        n = int(arr["selected_path"].shape[0])
        if rem < n:
            out = {k: arr[k][rem] for k in arr.files}
            out["source_chunk"] = np.array(c.name)
            return out
        rem -= n
    raise IndexError(f"sample_index={sample_index} out of range")


def run_inference(
    model: AdjusterDiffusionModel,
    schedule: DiffusionSchedule,
    selected_path: torch.Tensor,
    ctx_120: torch.Tensor,
    ctx_240: torch.Tensor,
    ctx_480: torch.Tensor,
    regime: torch.Tensor,
    quant: torch.Tensor,
    xgb: torch.Tensor,
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        future = selected_path + 0.1 * torch.randn_like(selected_path)
        bsz = selected_path.shape[0]
        max_steps = 50

        for i, t_scalar in enumerate(range(schedule.timesteps, 0, -1)):
            if i >= max_steps:
                break

            t = torch.full((bsz,), t_scalar, device=selected_path.device, dtype=torch.long)

            eps = model(
                noisy_future=future,
                selected_path=selected_path,
                ctx_120=ctx_120,
                ctx_240=ctx_240,
                ctx_480=ctx_480,
                regime=regime,
                quant=quant,
                xgb=xgb,
                t=t,
            )

            step_size = 0.05
            future = future - step_size * eps

        future = 0.75 * selected_path + 0.25 * future
        return future


def main() -> int:
    parser = argparse.ArgumentParser(description="Run adjuster diffusion inference on one sample")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="C:/PersonalDrive/Programming/AiStudio/nexus-trader/nexus_packaged/MMFPS/adjuster/checkpoints/adjuster_diffusion.pt",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="C:/PersonalDrive/Programming/AiStudio/nexus-trader/nexus_packaged/MMFPS/adjuster/dataset",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--num-samples", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    xgb_model = xgb.Booster()
    xgb_model.load_model("nexus_packaged/MMFPS/models/xgb_path_scorer.json")
    model = AdjusterDiffusionModel(
        horizon=int(ckpt["horizon"]),
        regime_dim=int(ckpt.get("regime_dim", 4)),
        quant_dim=int(ckpt.get("quant_dim", 4)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)

    schedule = DiffusionSchedule(
        DiffusionConfig(
            timesteps=int(ckpt["timesteps"]),
            beta_start=float(ckpt["beta_start"]),
            beta_end=float(ckpt["beta_end"]),
        )
    ).to(device)

    def extract_features(path: torch.Tensor):
        # path: (H,)
        mean = path.mean()
        std = path.std()
        max_val = path.max()
        min_val = path.min()
        range_ = max_val - min_val

        # simple drawdown approximation
        cummax = torch.cummax(path, dim=0).values
        drawdown = (path - cummax).min()

        return torch.stack([mean, std, max_val, min_val, range_, drawdown])

    all_results = []

    for i in range(args.num_samples):

        sample = _load_sample(Path(args.data_dir), i)

        selected = torch.from_numpy(sample["selected_path"]).float().unsqueeze(0).to(device)
        ctx_120 = torch.from_numpy(sample["ctx_120"]).float().unsqueeze(0).to(device)
        ctx_240 = torch.from_numpy(sample["ctx_240"]).float().unsqueeze(0).to(device)
        ctx_480 = torch.from_numpy(sample["ctx_480"]).float().unsqueeze(0).to(device)
        regime = torch.from_numpy(sample["regime"]).float().unsqueeze(0).to(device)
        quant = torch.from_numpy(sample["quant"]).float().unsqueeze(0).to(device)
        xgb_feat = torch.from_numpy(sample["xgb"]).float().reshape(1, 1).to(device)

        candidates = []

        for _ in range(8):  # K = 5
            pred = run_inference(
                model=model,
                schedule=schedule,
                selected_path=selected,
                ctx_120=ctx_120,
                ctx_240=ctx_240,
                ctx_480=ctx_480,
                regime=regime,
                quant=quant,
                xgb=xgb_feat,
            )
            candidates.append(pred)

        candidates = torch.cat(candidates, dim=0)  # (K, H)

        # simple scoring (baseline)
        
        features_list = []

        for c in candidates:
            f = extract_features(c)
            features_list.append(f)

        features = torch.stack(features_list).cpu().numpy()  # (K, F)

        # XGB scores
        dmat = xgb.DMatrix(features)
        scores = xgb_model.predict(dmat)
        scores = torch.from_numpy(scores).to(candidates.device)

        # --- HYBRID PART ---
        selected_exp = selected.expand(candidates.shape[0], -1)

        # deviation penalty (how far candidate moves from base path)
        deviation = ((candidates - selected_exp) ** 2).mean(dim=1).sqrt()
        vol = candidates.std(dim=1)
        base_vol = selected.std(dim=1).expand_as(vol)

        vol_ratio = vol / (base_vol + 1e-6)

       # Normalize everything
        scores_norm = (scores - scores.mean()) / (scores.std() + 1e-6)
        dev_norm = (deviation - deviation.mean()) / (deviation.std() + 1e-6)
        vol_norm = (vol_ratio - vol_ratio.mean()) / (vol_ratio.std() + 1e-6)

        # Final score (single consistent formula)
        lambda_penalty = 0.30

        final_score = (
            0.75 * scores_norm
            - 0.15 * dev_norm
            - 0.10 * vol_norm
        )

        # Apply mask BEFORE selection
        mask = (deviation < 0.13) & (vol_ratio < 1.8)

        if mask.sum() > 3:
            final_score = final_score.clone()
            final_score[~mask] = -1e9
        

        best_idx = torch.argmax(final_score)
        refined = candidates[best_idx].unsqueeze(0)

        selected_np = selected.squeeze(0).cpu().numpy()
        refined_np = refined.squeeze(0).cpu().numpy()
        target_np = sample["clean_future"].astype(np.float32)

        result = evaluate_adjustment(selected_np, refined_np, target_np)
        all_results.append(result)

        print(f"[{i}] improvement={result['improvement_pct']:.2f} "
            f"dir={result['direction_acc_pct']:.2f}")

    def avg(key):
        return np.mean([r[key] for r in all_results])

    def std(key):
        return np.std([r[key] for r in all_results])

    print("\n=== AGGREGATED RESULTS ===")
    print(f"improvement_pct: mean={avg('improvement_pct'):.3f}, std={std('improvement_pct'):.3f}")
    print(f"direction_acc:   mean={avg('direction_acc_pct'):.3f}")
    print(f"max_move_ratio:  mean={avg('max_move_ratio'):.3f}")
    print(f"volatility_err:  mean={avg('volatility_error'):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
