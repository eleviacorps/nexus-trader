"""Inference for diffusion-based adjuster."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
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
        delta = torch.randn_like(selected_path)
        bsz = selected_path.shape[0]
        for t_scalar in range(schedule.timesteps, 0, -1):
            t = torch.full((bsz,), t_scalar, device=selected_path.device, dtype=torch.long)
            eps = model(
                noisy_delta=delta,
                selected_path=selected_path,
                ctx_120=ctx_120,
                ctx_240=ctx_240,
                ctx_480=ctx_480,
                regime=regime,
                quant=quant,
                xgb=xgb,
                t=t,
            )
            delta = schedule.p_sample_step(delta, eps, t_scalar)

        refined = selected_path + delta
        refined = torch.clamp(refined, -0.15, 0.15)  # Mandatory hard clamp.
        return refined


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
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

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

    sample = _load_sample(Path(args.data_dir), args.sample_index)
    selected = torch.from_numpy(sample["selected_path"]).float().unsqueeze(0).to(device)
    ctx_120 = torch.from_numpy(sample["ctx_120"]).float().unsqueeze(0).to(device)
    ctx_240 = torch.from_numpy(sample["ctx_240"]).float().unsqueeze(0).to(device)
    ctx_480 = torch.from_numpy(sample["ctx_480"]).float().unsqueeze(0).to(device)
    regime = torch.from_numpy(sample["regime"]).float().unsqueeze(0).to(device)
    quant = torch.from_numpy(sample["quant"]).float().unsqueeze(0).to(device)
    xgb = torch.from_numpy(sample["xgb"]).float().reshape(1, 1).to(device)

    refined = run_inference(
        model=model,
        schedule=schedule,
        selected_path=selected,
        ctx_120=ctx_120,
        ctx_240=ctx_240,
        ctx_480=ctx_480,
        regime=regime,
        quant=quant,
        xgb=xgb,
    )

    selected_np = selected.squeeze(0).cpu().numpy()
    refined_np = refined.squeeze(0).cpu().numpy()
    target_np = selected_np + sample["clean_delta"].astype(np.float32)

    print(f"sample_index={args.sample_index} source_chunk={sample['source_chunk'].item()}")
    result = evaluate_adjustment(selected_np, refined_np, target_np)
    print_metrics(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
