"""Train Selector v2 with mandatory staged rollout.

Order enforced:
1) Base scorer (no diffusion, no HMM/XGB)
2) Verify base training
3) Diffusion refinement (still no HMM/XGB)
4) Full model with HMM + XGBoost signals
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_p = Path(__file__).resolve().parents[2]
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))

from MMFPS.selector.selector_v2 import SelectorLoss, SelectorV2


class SelectorDataset(Dataset):
    """Preloads existing selector chunks; dataset format remains unchanged."""

    def __init__(self, data_dir: Path, num_samples: int = 40000, max_paths: int = 128):
        self.data_dir = Path(data_dir)
        self.max_paths = int(max_paths)
        self.num_samples = int(num_samples)

        chunk_files = sorted(self.data_dir.glob("chunk_*.npz"))
        if not chunk_files:
            raise FileNotFoundError(f"No chunk_*.npz files found in {self.data_dir}")

        print(f"Loading {len(chunk_files)} chunks...")
        contexts: list[np.ndarray] = []
        paths: list[np.ndarray] = []
        futures: list[np.ndarray] = []

        for chunk in chunk_files:
            arr = np.load(chunk)
            contexts.append(arr["contexts"])
            paths.append(arr["paths"])
            futures.append(arr["futures"])
            print(f"  {chunk.name}: {arr['contexts'].shape[0]}")

        self.contexts = np.concatenate(contexts, axis=0).astype(np.float32)
        self.paths = np.concatenate(paths, axis=0).astype(np.float32)
        self.futures = np.concatenate(futures, axis=0).astype(np.float32)

        print(f"Total samples: {self.contexts.shape[0]}")

    def __len__(self) -> int:
        return min(self.num_samples, len(self.contexts))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        context = self.contexts[idx]
        paths = self.paths[idx]
        real = self.futures[idx]

        if paths.shape[0] > self.max_paths:
            sel = np.random.choice(paths.shape[0], self.max_paths, replace=False)
            paths = paths[sel]

        return {
            "context": torch.from_numpy(context.copy()),
            "paths": torch.from_numpy(paths.copy()),
            "real": torch.from_numpy(real.copy()),
        }


@dataclass
class EpochMetrics:
    loss: float
    loss_soft: float
    loss_rank: float
    loss_pair_margin: float
    loss_topk: float
    loss_gap: float
    loss_var: float
    loss_energy: float
    loss_entropy: float
    score_std: float
    non_uniform: float
    topk_gap: float
    spearman: float
    topk_stability: float
    nan_steps: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_phase(
    model: SelectorV2,
    dataloader: DataLoader,
    criterion: SelectorLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    epochs: int,
    phase_name: str,
    soft_weight_start: float,
    soft_weight_end: float,
    input_noise_std: float,
    collapse_std_threshold: float,
    reset_optimizer_on_collapse: bool,
) -> list[EpochMetrics]:
    history: list[EpochMetrics] = []

    model.train()
    for epoch in range(epochs):
        if hasattr(criterion, "set_soft_weight"):
            if epochs <= 1:
                criterion.set_soft_weight(soft_weight_end)
            else:
                frac = epoch / float(epochs - 1)
                criterion.set_soft_weight(
                    soft_weight_start + (soft_weight_end - soft_weight_start) * frac
                )
        if hasattr(criterion, "set_epoch"):
            criterion.set_epoch(epoch)

        total_loss = 0.0
        total_loss_soft = 0.0
        total_loss_rank = 0.0
        total_loss_pair_margin = 0.0
        total_loss_topk = 0.0
        total_loss_gap = 0.0
        total_loss_var = 0.0
        total_loss_energy = 0.0
        total_loss_entropy = 0.0
        total_score_std = 0.0
        total_non_uniform = 0.0
        total_topk_gap = 0.0
        total_spearman = 0.0
        total_topk_stability = 0.0
        stability_steps = 0
        collapse_events = 0
        nan_steps = 0
        prev_topk_hist: torch.Tensor | None = None

        pbar = tqdm(dataloader, desc=f"{phase_name} | epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            context = batch["context"].to(device, non_blocking=True)
            paths = batch["paths"].to(device, non_blocking=True)
            real = batch["real"].to(device, non_blocking=True)
            if input_noise_std > 0.0:
                context = context + torch.randn_like(context) * input_noise_std

            optimizer.zero_grad(set_to_none=True)

            output = model(context, paths)
            loss_dict = criterion(output, paths, real)
            loss = loss_dict["loss"]

            if not torch.isfinite(loss):
                nan_steps += 1
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += float(loss.detach().item())
            total_loss_soft += float(loss_dict.get("loss_soft", loss).detach().item())
            total_loss_rank += float(loss_dict.get("loss_rank", torch.tensor(0.0, device=device)).detach().item())
            total_loss_pair_margin += float(loss_dict.get("loss_pair_margin", torch.tensor(0.0, device=device)).detach().item())
            total_loss_topk += float(loss_dict.get("loss_topk", torch.tensor(0.0, device=device)).detach().item())
            total_loss_gap += float(loss_dict.get("loss_gap", torch.tensor(0.0, device=device)).detach().item())
            total_loss_var += float(loss_dict.get("loss_var", torch.tensor(0.0, device=device)).detach().item())
            total_loss_energy += float(loss_dict.get("loss_energy", torch.tensor(0.0, device=device)).detach().item())
            total_loss_entropy += float(loss_dict.get("loss_entropy", torch.tensor(0.0, device=device)).detach().item())
            total_score_std += float(loss_dict["score_std"].detach().item())
            total_non_uniform += float(loss_dict["non_uniform"].detach().item())
            total_topk_gap += float(loss_dict.get("topk_gap", torch.tensor(0.0, device=device)).detach().item())
            total_spearman += float(loss_dict.get("spearman", torch.tensor(0.0, device=device)).detach().item())

            topk = min(int(getattr(criterion, "topk", 10)), output["final_scores"].shape[1])
            pred_top_idx = torch.topk(output["final_scores"].detach(), k=topk, dim=1).indices
            hist = torch.bincount(
                pred_top_idx.reshape(-1),
                minlength=output["final_scores"].shape[1],
            ).float()
            hist = hist / hist.sum().clamp(min=1e-6)
            if prev_topk_hist is not None:
                stability = F.cosine_similarity(
                    hist.unsqueeze(0), prev_topk_hist.unsqueeze(0), dim=-1
                )[0]
                total_topk_stability += float(stability.item())
                stability_steps += 1
            prev_topk_hist = hist

            batch_std = float(loss_dict["score_std"].detach().item())
            if batch_std < collapse_std_threshold:
                collapse_events += 1
                if reset_optimizer_on_collapse:
                    optimizer = optim.AdamW(
                        model.parameters(),
                        lr=optimizer.param_groups[0]["lr"],
                        weight_decay=optimizer.param_groups[0]["weight_decay"],
                    )
                    pbar.write(
                        f"[{phase_name}] collapse_detected std={batch_std:.5f}; optimizer RESET."
                    )

            pbar.set_postfix(
                loss=f"{float(loss.detach().item()):.6f}",
                soft=f"{float(loss_dict.get('loss_soft', loss).detach().item()):.6f}",
                rank=f"{float(loss_dict.get('loss_rank', torch.tensor(0.0, device=device)).detach().item()):.5f}",
                gap=f"{float(loss_dict.get('topk_gap', torch.tensor(0.0, device=device)).detach().item()):.4f}",
                std=f"{float(loss_dict['score_std'].detach().item()):.4f}",
            )

        n_batches = max(len(dataloader), 1)
        metrics = EpochMetrics(
            loss=total_loss / n_batches,
            loss_soft=total_loss_soft / n_batches,
            loss_rank=total_loss_rank / n_batches,
            loss_pair_margin=total_loss_pair_margin / n_batches,
            loss_topk=total_loss_topk / n_batches,
            loss_gap=total_loss_gap / n_batches,
            loss_var=total_loss_var / n_batches,
            loss_energy=total_loss_energy / n_batches,
            loss_entropy=total_loss_entropy / n_batches,
            score_std=total_score_std / n_batches,
            non_uniform=total_non_uniform / n_batches,
            topk_gap=total_topk_gap / n_batches,
            spearman=total_spearman / n_batches,
            topk_stability=(total_topk_stability / max(stability_steps, 1)),
            nan_steps=nan_steps,
        )
        history.append(metrics)

        print(
            f"[{phase_name}] epoch={epoch + 1} "
            f"loss={metrics.loss:.6f} soft={metrics.loss_soft:.6f} "
            f"rank={metrics.loss_rank:.6f} pair_m={metrics.loss_pair_margin:.6f} "
            f"topk={metrics.loss_topk:.6f} gap_l={metrics.loss_gap:.6f} var={metrics.loss_var:.6f} "
            f"energy={metrics.loss_energy:.6f} ent={metrics.loss_entropy:.6f} "
            f"score_std={metrics.score_std:.4f} gap={metrics.topk_gap:.4f} "
            f"spr={metrics.spearman:.4f} stab={metrics.topk_stability:.4f} "
            f"non_uniform={metrics.non_uniform:.3f} collapses={collapse_events} "
            f"nan_steps={metrics.nan_steps}"
        )

    return history


def verify_base_phase(history: list[EpochMetrics]) -> None:
    if not history:
        raise RuntimeError("Base phase did not run any epochs.")

    first = history[0].loss
    last = history[-1].loss
    best = min(h.loss for h in history)
    any_nan = any(h.nan_steps > 0 for h in history)
    final_non_uniform = history[-1].non_uniform

    if any_nan:
        raise RuntimeError("Base phase produced NaN/Inf steps; aborting before diffusion.")
    if not (best < first and last <= first * 1.05):
        raise RuntimeError(
            "Base phase verification failed "
            f"(first={first:.6f}, best={best:.6f}, last={last:.6f})."
        )
    if final_non_uniform <= 0.5:
        raise RuntimeError(
            f"Base phase verification failed (scores too uniform: non_uniform={final_non_uniform:.3f})."
        )



def build_model(
    max_paths: int,
    use_diffusion: bool,
    use_hmm: bool,
    use_xgboost: bool,
    diffusion_steps: int,
    hmm_model_path: str | None,
    xgb_model_path: str | None,
) -> SelectorV2:
    return SelectorV2(
        feature_dim=144,
        path_len=20,
        num_paths=max_paths,
        d_model=256,
        num_heads=8,
        num_ctx_layers=2,
        use_diffusion=use_diffusion,
        diffusion_steps=diffusion_steps,
        num_regimes=6,
        use_hmm=use_hmm,
        use_xgboost=use_xgboost,
        hmm_model_path=hmm_model_path,
        xgb_model_path=xgb_model_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train MMFPS Selector v2")
    parser.add_argument("--data-dir", type=str, default="C:/PersonalDrive/Programming/AiStudio/nexus-trader/nexus_packaged/outputs/MMFPS/selector_data")
    parser.add_argument("--output-dir", type=str, default="C:/PersonalDrive/Programming/AiStudio/nexus-trader/nexus_packaged/outputs/MMFPS/selector_checkpoints_v2")

    parser.add_argument("--num-samples", type=int, default=40000)
    parser.add_argument("--max-paths", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--base-epochs", type=int, default=5)
    parser.add_argument("--diffusion-epochs", type=int, default=4)
    parser.add_argument("--full-epochs", type=int, default=8)
    parser.add_argument("--diffusion-steps", type=int, default=4)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk-weight", type=float, default=0.5)
    parser.add_argument("--rank-weight", type=float, default=1.0)
    parser.add_argument("--var-weight", type=float, default=0.1)
    parser.add_argument("--energy-weight", type=float, default=0.15)
    parser.add_argument("--gap-weight", type=float, default=0.1)
    parser.add_argument("--entropy-weight", type=float, default=0.1)
    parser.add_argument("--soft-weight-start", type=float, default=0.05)
    parser.add_argument("--soft-weight-end", type=float, default=0.01)
    parser.add_argument("--rank-temp", type=float, default=0.3)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--pair-margin", type=float, default=0.05)
    parser.add_argument("--pair-margin-weight", type=float, default=0.2)
    parser.add_argument("--topk-margin-alpha", type=float, default=0.5)
    parser.add_argument("--min-std", type=float, default=0.2)
    parser.add_argument("--target-gap", type=float, default=1.0)
    parser.add_argument("--entropy-target", type=float, default=1.2)
    parser.add_argument("--entropy-warmup-epochs", type=int, default=2)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--input-noise-std", type=float, default=0.002)
    parser.add_argument("--collapse-std-threshold", type=float, default=0.05)
    parser.add_argument("--reset-optimizer-on-collapse", action="store_true")

    parser.add_argument("--hmm-model-path", type=str, default=None)
    parser.add_argument("--xgb-model-path", type=str, default=None)
    parser.add_argument("--resume-base", action="store_true")
    parser.add_argument("--resume-base-checkpoint", type=str, default=None)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"device={device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = SelectorDataset(
        data_dir=Path(args.data_dir),
        num_samples=args.num_samples,
        max_paths=args.max_paths,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    print(f"batches={len(dataloader)}")

    criterion = SelectorLoss(
        rank_weight=args.rank_weight,
        topk_weight=args.topk_weight,
        variance_weight=args.var_weight,
        energy_weight=args.energy_weight,
        gap_weight=args.gap_weight,
        entropy_weight=args.entropy_weight,
        soft_weight=args.soft_weight_start,
        rank_temp=args.rank_temp,
        margin=args.margin,
        pair_margin=args.pair_margin,
        pair_margin_weight=args.pair_margin_weight,
        topk_margin_alpha=args.topk_margin_alpha,
        min_std=args.min_std,
        target_gap=args.target_gap,
        entropy_target=args.entropy_target,
        entropy_warmup_epochs=args.entropy_warmup_epochs,
        topk=args.topk,
    )

    # Step 1-6: base scorer only.
    base_model = build_model(
        max_paths=args.max_paths,
        use_diffusion=False,
        use_hmm=False,
        use_xgboost=False,
        diffusion_steps=args.diffusion_steps,
        hmm_model_path=args.hmm_model_path,
        xgb_model_path=args.xgb_model_path,
    ).to(device)

    print(f"base_params_m={sum(p.numel() for p in base_model.parameters()) / 1e6:.2f}")
    base_ckpt: Path | None = None
    if args.resume_base_checkpoint:
        base_ckpt = Path(args.resume_base_checkpoint)
    elif args.resume_base:
        base_ckpt = output_dir / "selector_v2_base.pt"

    if base_ckpt is not None:
        if not base_ckpt.exists():
            raise FileNotFoundError(f"Requested base checkpoint not found: {base_ckpt}")
        state = torch.load(base_ckpt, map_location=device, weights_only=True)
        base_model.load_state_dict(state, strict=False)
        print(f"loaded_base_checkpoint={base_ckpt}")
    else:
        base_optim = optim.AdamW(base_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        base_hist = train_phase(
            model=base_model,
            dataloader=dataloader,
            criterion=criterion,
            optimizer=base_optim,
            device=device,
            epochs=args.base_epochs,
            phase_name="base",
            soft_weight_start=args.soft_weight_start,
            soft_weight_end=args.soft_weight_end,
            input_noise_std=args.input_noise_std,
            collapse_std_threshold=args.collapse_std_threshold,
            reset_optimizer_on_collapse=args.reset_optimizer_on_collapse,
        )
        verify_base_phase(base_hist)
        torch.save(base_model.state_dict(), output_dir / "selector_v2_base.pt")

    current_model = base_model

    # Step 7: add diffusion.
    if args.diffusion_epochs > 0:
        diff_model = build_model(
            max_paths=args.max_paths,
            use_diffusion=True,
            use_hmm=False,
            use_xgboost=False,
            diffusion_steps=args.diffusion_steps,
            hmm_model_path=args.hmm_model_path,
            xgb_model_path=args.xgb_model_path,
        ).to(device)

        diff_model.load_state_dict(current_model.state_dict(), strict=False)
        diff_optim = optim.AdamW(diff_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        _ = train_phase(
            model=diff_model,
            dataloader=dataloader,
            criterion=criterion,
            optimizer=diff_optim,
            device=device,
            epochs=args.diffusion_epochs,
            phase_name="diffusion",
            soft_weight_start=args.soft_weight_start,
            soft_weight_end=args.soft_weight_end,
            input_noise_std=args.input_noise_std,
            collapse_std_threshold=args.collapse_std_threshold,
            reset_optimizer_on_collapse=args.reset_optimizer_on_collapse,
        )
        torch.save(diff_model.state_dict(), output_dir / "selector_v2_diffusion.pt")
        current_model = diff_model

    # Step 8-9: enable HMM + XGBoost features.
    if args.full_epochs > 0:
        full_model = build_model(
            max_paths=args.max_paths,
            use_diffusion=True,
            use_hmm=True,
            use_xgboost=True,
            diffusion_steps=args.diffusion_steps,
            hmm_model_path=args.hmm_model_path,
            xgb_model_path=args.xgb_model_path,
        ).to(device)

        full_model.load_state_dict(current_model.state_dict(), strict=False)
        full_optim = optim.AdamW(full_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        _ = train_phase(
            model=full_model,
            dataloader=dataloader,
            criterion=criterion,
            optimizer=full_optim,
            device=device,
            epochs=args.full_epochs,
            phase_name="full",
            soft_weight_start=args.soft_weight_start,
            soft_weight_end=args.soft_weight_end,
            input_noise_std=args.input_noise_std,
            collapse_std_threshold=args.collapse_std_threshold,
            reset_optimizer_on_collapse=args.reset_optimizer_on_collapse,
        )
        torch.save(full_model.state_dict(), output_dir / "selector_v2_full.pt")
        current_model = full_model

    torch.save(current_model.state_dict(), output_dir / "selector_v2_latest.pt")
    print(f"saved={output_dir / 'selector_v2_latest.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
