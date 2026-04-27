"""Train selector - FIXED data pipeline (preload into RAM)."""

import sys
from pathlib import Path
_p = Path(__file__).resolve().parents[2]
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.backends.cudnn import benchmark
from tqdm import tqdm

from MMFPS.selector.hybrid_selector import HybridIntelligenceSelector


class SelectorDataset(Dataset):
    """FULL PRELOAD - load ALL chunks into RAM once."""

    def __init__(self, data_dir, num_samples=40000, max_paths=128):
        self.data_dir = Path(data_dir)
        self.max_paths = max_paths
        self.num_samples = num_samples

        chunk_files = sorted(self.data_dir.glob("chunk_*.npz"))
        print(f"Loading {len(chunk_files)} chunks into RAM...")

        contexts = []
        paths = []
        futures = []

        for f in chunk_files:
            d = np.load(f)
            contexts.append(d["contexts"])
            paths.append(d["paths"])
            futures.append(d["futures"])
            print(f"  Loaded {f.name}: {d['contexts'].shape[0]} samples")

        self.contexts = np.concatenate(contexts, axis=0).astype(np.float32)
        self.paths = np.concatenate(paths, axis=0).astype(np.float32)
        self.futures = np.concatenate(futures, axis=0).astype(np.float32)

        print(f"Total: {self.contexts.shape[0]} samples, {self.paths.shape[0]} total paths")
        print(f"Memory: contexts={self.contexts.nbytes/1e9:.1f}GB, paths={self.paths.nbytes/1e9:.1f}GB")

    def __len__(self):
        return min(self.num_samples, len(self.contexts))

    def __getitem__(self, idx):
        ctx = self.contexts[idx]
        path = self.paths[idx]
        real = self.futures[idx]

        if path.shape[0] > self.max_paths:
            sel = np.random.choice(path.shape[0], self.max_paths, replace=False)
            path = path[sel]

        return {
            'context': torch.from_numpy(ctx.copy()),
            'paths': torch.from_numpy(path.copy()),
            'real': torch.from_numpy(real.copy()),
        }


class SimpleLoss(nn.Module):
    def forward(self, output, actual_path):
        # For price changes, compute return as final - initial
        actual_ret = (actual_path[:, -1] - actual_path[:, 0])
        mse = nn.functional.mse_loss(output.expected_return, actual_ret)
        dir_pred = (output.expected_return > 0).float()
        dir_actual = (actual_ret > 0).float()
        bce = nn.functional.binary_cross_entropy_with_logits(dir_pred, dir_actual)
        return mse + 0.5 * bce


def train():
    data_dir = Path("C:/PersonalDrive/Programming/AiStudio/nexus-trader/nexus_packaged/outputs/MMFPS/selector_data")
    output_dir = Path("C:/PersonalDrive/Programming/AiStudio/nexus-trader/nexus_packaged/outputs/MMFPS/selector_checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(True)

    BATCH = 64
    EPOCHS = 10
    LR = 1e-3
    NUM_SAMPLES = 40000
    MAX_PATHS = 128

    dataset = SelectorDataset(data_dir, num_samples=NUM_SAMPLES, max_paths=MAX_PATHS)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    print(f"\nBatches: {len(dataloader)}")

    model = HybridIntelligenceSelector(
        feature_dim=144,
        path_len=20,
        num_paths=MAX_PATHS,
        d_model=256,
        num_heads=8,
        num_gru_layers=2,
        use_diffusion=False,
        use_xgboost=False,
    ).to(device)

    print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    criterion = SimpleLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch in pbar:
            ctx = batch['context'].to(device, non_blocking=True)
            paths = batch['paths'].to(device, non_blocking=True)
            real = batch['real'].to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.autocast("cuda", torch.float16):
                output = model(ctx, paths, real)

            loss = criterion(output, real)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}")

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), output_dir / f"checkpoint_{epoch+1}.pt")

    torch.save(model.state_dict(), output_dir / "best_model.pt")
    print(f"Done. Saved to {output_dir}")


if __name__ == "__main__":
    train()