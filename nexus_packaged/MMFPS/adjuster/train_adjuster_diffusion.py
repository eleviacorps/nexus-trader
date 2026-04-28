"""Train diffusion-based residual adjuster."""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

_p = Path(__file__).resolve().parents[2]
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))

from MMFPS.adjuster.adjuster_diffusion_model import (  # noqa: E402
    AdjusterDiffusionModel,
    DiffusionConfig,
    DiffusionSchedule,
)


class AdjusterDataset(Dataset):
    def __init__(self, data_dir: Path, num_samples: int = 0):
        self.data_dir = Path(data_dir)
        chunks = sorted(self.data_dir.glob("chunk_*.npz"))
        if not chunks:
            chunks = sorted(self.data_dir.glob("*.npz"))
        chunks = [c for c in chunks if c.name != "meta.npz"]
        if not chunks:
            raise FileNotFoundError(f"No adjuster dataset chunks found in {self.data_dir}")

        fields = [
            "noisy_future",
            "clean_future",
            "t",
            "selected_path",
            "ctx_120",
            "ctx_240",
            "ctx_480",
            "regime",
            "quant",
            "xgb",
        ]
        store: dict[str, list[np.ndarray]] = {k: [] for k in fields}

        print(f"Loading {len(chunks)} adjuster chunks...")
        for c in chunks:
            arr = np.load(c)
            for k in fields:
                store[k].append(arr[k])
            print(f"  {c.name}: {arr['noisy_future'].shape[0]}")

        self.data: dict[str, np.ndarray] = {
            k: np.concatenate(v, axis=0).astype(np.float32 if k != "t" else np.int64)
            for k, v in store.items()
        }

        total = self.data["noisy_future"].shape[0]
        if num_samples > 0:
            keep = min(num_samples, total)
            for k in self.data:
                self.data[k] = self.data[k][:keep]
            total = keep
        print(f"Total adjuster samples: {total}")

    def __len__(self) -> int:
        return int(self.data["noisy_future"].shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "noisy_future": torch.from_numpy(self.data["noisy_future"][idx]),
            "clean_future": torch.from_numpy(self.data["clean_future"][idx]),
            "t": torch.tensor(self.data["t"][idx], dtype=torch.long),
            "selected_path": torch.from_numpy(self.data["selected_path"][idx]),
            "ctx_120": torch.from_numpy(self.data["ctx_120"][idx]),
            "ctx_240": torch.from_numpy(self.data["ctx_240"][idx]),
            "ctx_480": torch.from_numpy(self.data["ctx_480"][idx]),
            "regime": torch.from_numpy(self.data["regime"][idx]),
            "quant": torch.from_numpy(self.data["quant"][idx]),
            "xgb": torch.from_numpy(self.data["xgb"][idx]),
        }


@dataclass
class EpochStats:
    loss: float
    loss_diff: float
    loss_recon: float
    loss_dir: float
    loss_mag: float
    step_ratio_mean: float
    loss_vel: float
    loss_accel: float
    nan_steps: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def _epoch_pass(
    model: AdjusterDiffusionModel,
    schedule: DiffusionSchedule,
    loader: DataLoader,
    device: torch.device,
    optimizer: optim.Optimizer | None,
    sampling_step_size: float,
) -> EpochStats:
    train = optimizer is not None
    model.train(mode=train)
    total_loss = 0.0
    total_diff = 0.0
    total_recon = 0.0
    total_dir = 0.0
    total_mag = 0.0
    total_step_ratio = 0.0
    total_vel = 0.0
    total_accel = 0.0
    nan_steps = 0

    iterator = tqdm(loader, leave=False)
    for batch in iterator:
        batch = _move_batch(batch, device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        eps_pred = model(
            noisy_future=batch["noisy_future"],
            selected_path=batch["selected_path"],
            ctx_120=batch["ctx_120"],
            ctx_240=batch["ctx_240"],
            ctx_480=batch["ctx_480"],
            regime=batch["regime"],
            quant=batch["quant"],
            xgb=batch["xgb"],
            t=batch["t"],
        )

        # noise_target = (x_t - sqrt(alpha_bar_t) * x_0) / sqrt(1 - alpha_bar_t)
        sqrt_ab = schedule.extract(schedule.alpha_bars.sqrt(), batch["t"], batch["clean_future"].ndim)
        sqrt_1m = schedule.extract((1.0 - schedule.alpha_bars).sqrt(), batch["t"], batch["clean_future"].ndim)
        noise_target = (batch["noisy_future"] - sqrt_ab * batch["clean_future"]) / (sqrt_1m + 1e-8)

        loss_diff = F.mse_loss(eps_pred, noise_target)
        # Reconstruct predicted residual delta from epsilon prediction.
        selected = batch["selected_path"]
        clean = batch["clean_future"]

        alpha_bar_t = schedule.extract(schedule.alpha_bars, batch["t"], batch["noisy_future"].ndim)

        pred_future = (
            batch["noisy_future"] - torch.sqrt(1 - alpha_bar_t) * eps_pred
        ) / torch.sqrt(alpha_bar_t + 1e-8)

        target_future = batch["clean_future"]

        # ✅ ADD THIS (train = inference alignment)
        pred_future = torch.clamp(pred_future, -0.30, 0.30)

        # ✅ ADD THIS (final objective)
        loss_mse_final = F.mse_loss(pred_future, target_future)

        # ✅ ADD THIS (improvement constraint)
        mse_before = F.mse_loss(selected, target_future)
        mse_after  = F.mse_loss(pred_future, target_future)
        loss_improve = torch.relu(mse_after - mse_before)

        loss_mse_final = F.mse_loss(pred_future, target_future)

        # Sampling-consistency step: enforce behavior on generation-like update.

        loss_recon = F.mse_loss(pred_future, target_future)

        loss_dir = 1 - F.cosine_similarity(pred_future, target_future, dim=-1).mean()

        ratio = pred_future.abs().mean() / (target_future.abs().mean() + 1e-6)
        loss_mag = torch.relu(0.9 - ratio)

        vel = pred_future[:, 1:] - pred_future[:, :-1]
        acc = vel[:, 1:] - vel[:, :-1]

        loss_vel = vel.pow(2).mean()
        loss_accel = acc.pow(2).mean()

        loss = (
            loss_diff
            + 0.3 * loss_recon
            + 1.2 * loss_dir
            + 1.5 * loss_mag
            + 0.05 * loss_vel
            + 0.03 * loss_accel
        )

        if not torch.isfinite(loss):
            nan_steps += 1
            if train:
                optimizer.zero_grad(set_to_none=True)
            continue

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += float(loss.detach().item())
        total_diff += float(loss_diff.detach().item())
        total_recon += float(loss_recon.detach().item())
        total_dir += float(loss_dir.detach().item())
        total_mag += float(loss_mag.detach().item())
        total_step_ratio += float(ratio.mean().detach().item())
        total_vel += float(loss_vel.detach().item())
        total_accel += float(loss_accel.detach().item())

        iterator.set_postfix(
            loss=f"{loss.item():.5f}",
            diff=f"{loss_diff.item():.5f}",
            recon=f"{loss_recon.item():.5f}",
            dir=f"{loss_dir.item():.5f}",
            mag=f"{loss_mag.item():.5f}",
            step_ratio=f"{ratio.mean().item():.4f}",
        )

    n = max(len(loader) - nan_steps, 1)
    return EpochStats(
        loss=total_loss / n,
        loss_diff=total_diff / n,
        loss_recon=total_recon / n,
        loss_dir=total_dir / n,
        loss_mag=total_mag / n,
        step_ratio_mean=total_step_ratio / n,
        loss_vel=total_vel / n,
        loss_accel=total_accel / n,
        nan_steps=nan_steps,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train adjuster diffusion model")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="C:/PersonalDrive/Programming/AiStudio/nexus-trader/nexus_packaged/MMFPS/adjuster/dataset",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="C:/PersonalDrive/Programming/AiStudio/nexus-trader/nexus_packaged/MMFPS/adjuster/checkpoints/adjuster_diffusion.pt",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=2e-2)
    parser.add_argument("--num-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampling-step-size", type=float, default=0.1)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"device={device}")

    data_dir = Path(args.data_dir)
    ckpt_path = Path(args.checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    # If dataset meta is present, use its diffusion schedule by default.
    meta = data_dir / "meta.npz"
    timesteps = int(args.timesteps)
    beta_start = float(args.beta_start)
    beta_end = float(args.beta_end)
    if meta.exists():
        m = np.load(meta)
        timesteps = int(m["timesteps"][0])
        beta_start = float(m["beta_start"][0])
        beta_end = float(m["beta_end"][0])
        print(f"Loaded schedule from {meta}: timesteps={timesteps}, beta=[{beta_start}, {beta_end}]")

    dataset = AdjusterDataset(data_dir=data_dir, num_samples=args.num_samples)
    n_train = int(0.9 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    print(f"train_batches={len(train_loader)} val_batches={len(val_loader)}")

    horizon = int(dataset.data["selected_path"].shape[1])
    regime_dim = int(dataset.data["regime"].shape[1])
    quant_dim = int(dataset.data["quant"].shape[1])

    model = AdjusterDiffusionModel(
        horizon=horizon,
        regime_dim=regime_dim,
        quant_dim=quant_dim,
    ).to(device)
    params_m = model.count_parameters() / 1e6
    print(f"params_m={params_m:.3f}")
    if params_m > 15.0:
        raise RuntimeError(f"Model exceeds 15M parameter constraint: {params_m:.3f}M")

    schedule = DiffusionSchedule(
        DiffusionConfig(timesteps=timesteps, beta_start=beta_start, beta_end=beta_end)
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val = float("inf")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        tr = _epoch_pass(
            model,
            schedule,
            train_loader,
            device,
            optimizer=optimizer,
            sampling_step_size=args.sampling_step_size,
        )
        va = _epoch_pass(
            model,
            schedule,
            val_loader,
            device,
            optimizer=None,
            sampling_step_size=args.sampling_step_size,
        )
        print(
            f"train: loss={tr.loss:.6f} diff={tr.loss_diff:.6f} recon={tr.loss_recon:.6f} "
            f"dir={tr.loss_dir:.6f} mag={tr.loss_mag:.6f} step_ratio_mean={tr.step_ratio_mean:.4f} "
            f"vel={tr.loss_vel:.6f} acc={tr.loss_accel:.6f} "
            f"nan={tr.nan_steps} | "
            f"val: loss={va.loss:.6f} diff={va.loss_diff:.6f} recon={va.loss_recon:.6f} "
            f"dir={va.loss_dir:.6f} mag={va.loss_mag:.6f} step_ratio_mean={va.step_ratio_mean:.4f} "
            f"vel={va.loss_vel:.6f} acc={va.loss_accel:.6f} "
            f"nan={va.nan_steps}"
        )
        print(f"step_ratio_mean={tr.step_ratio_mean:.4f}")

        state = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch + 1,
            "timesteps": timesteps,
            "beta_start": beta_start,
            "beta_end": beta_end,
            "horizon": horizon,
            "regime_dim": regime_dim,
            "quant_dim": quant_dim,
        }
        torch.save(state, ckpt_path)
        if va.loss < best_val:
            best_val = va.loss
            best_path = ckpt_path.with_name("adjuster_diffusion_best.pt")
            torch.save(state, best_path)
            print(f"saved_best={best_path}")

    print(f"saved={ckpt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
