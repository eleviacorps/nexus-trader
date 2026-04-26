import torch, json, sys, numpy as np, pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, ".")
from v30.models.selector.distribution_selector import DistributionSelector, DistributionLoss
from v30.models.selector.diffusion_selector import DiffusionSelector, DiffusionLoss


class HybridDataset(Dataset):
    def __init__(self, features, paths, close_prices, num_samples=5000, start_idx=30000):
        self.features = features
        self.paths = paths
        self.close_prices = close_prices
        self.num_samples = num_samples
        self.start_idx = start_idx

    def __len__(self):
        return min(self.num_samples, len(self.paths) - self.start_idx)

    def __getitem__(self, idx):
        path_idx = self.start_idx + idx
        context = self.features[path_idx * 15].astype(np.float32)
        paths = self.paths[path_idx].astype(np.float32)
        actual_start = path_idx * 15
        actual_indices = [actual_start + i * 15 for i in range(21)]
        actual = np.array([self.close_prices[min(i, len(self.close_prices) - 1)] for i in actual_indices], dtype=np.float32)
        return {"context": context, "paths": paths, "actual": actual}


project_root = Path(__file__).resolve().parents[3]  # nexus_packaged/v30/training -> nexus_packaged -> nexus-packaged -> nexus-trader -> project_root

features = np.load(project_root / "data/features/diffusion_fused_6m.npy").astype(np.float32)
ohlcv_df = pd.read_parquet(project_root / "nexus_packaged/data/ohlcv.parquet")
close_prices = ohlcv_df["close"].values.astype(np.float32)
paths = np.load(Path(__file__).resolve().parents[1] / "data/processed/v30_paths.npy").astype(np.float32)

val_ds = HybridDataset(features, paths, close_prices, 5000, 30000)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

device = torch.device("cuda")

output_dir = Path(__file__).resolve().parents[1] / "models/selector"

dist_model = DistributionSelector(144, 20, 64, 128, 0.1).to(device)
diff_model = DiffusionSelector(144, 20, 64, 128, 50, 0.1).to(device)

dist_ckpt = torch.load(output_dir / "best_distribution_selector.pt", weights_only=True)
diff_ckpt = torch.load(output_dir / "best_diffusion_selector.pt", weights_only=True)

dist_model.load_state_dict(dist_ckpt["model_state_dict"])
diff_model.load_state_dict(diff_ckpt["model_state_dict"])

dist_loss_fn = DistributionLoss(1.0, 0.5, 0.01)
diff_loss_fn = DiffusionLoss(1.0, 0.5, 0.01, 0.3, 0.1, 0.1, 0.05)


def validate(model, loader, loss_fn, is_diff):
    model.eval()
    total = {}
    n = 0
    with torch.no_grad():
        for batch in loader:
            ctx = batch["context"].to(device)
            p = batch["paths"].to(device)
            a = batch["actual"].to(device)
            if is_diff:
                out = model(ctx, p, return_noised=False)
                m = loss_fn(out, a)
            else:
                out = model(ctx, p)
                m = loss_fn(out, p, a)
            for k, v in m.items():
                total[k] = total.get(k, 0) + v.item()
            n += 1
    return {k: v / n for k, v in total.items()}


dv = validate(dist_model, val_loader, dist_loss_fn, False)
dfv = validate(diff_model, val_loader, diff_loss_fn, True)

print("=" * 60)
print(f"DIST (best checkpoint):  corr={dv['corr_with_actual']:.4f}  dir_acc={dv['dir_accuracy']:.4f}")
print(f"DIFFUSION (best chkpt):  corr={dfv['corr_with_actual']:.4f}  dir_acc={dfv['dir_accuracy']:.4f}")
print(f"Corr improvement:        {dfv['corr_with_actual'] - dv['corr_with_actual']:+.4f} ({((dfv['corr_with_actual'] - dv['corr_with_actual']) / (dv['corr_with_actual'] + 1e-6)) * 100:+.1f}%)")
print(f"Dir improvement:         {dfv['dir_accuracy'] - dv['dir_accuracy']:+.4f}")
print("=" * 60)
print(f"corr >= 0.30:  {dfv['corr_with_actual'] >= 0.30}")
print(f"dir >= 0.61:   {dfv['dir_accuracy'] >= 0.61}")
adopt = dfv["corr_with_actual"] >= 0.30 and dfv["dir_accuracy"] >= 0.61
print(f"VERDICT:      {'ADOPT DIFFUSION_SELECTOR' if adopt else 'KEEP DISTRIBUTION_SELECTOR'}")

output_dir = Path(__file__).resolve().parents[1] / "models/selector"
summary = {
    "best_dist_corr": dv["corr_with_actual"],
    "best_diff_corr": dfv["corr_with_actual"],
    "dist_dir_acc": dv["dir_accuracy"],
    "diff_dir_acc": dfv["dir_accuracy"],
    "corr_improvement_pct": ((dfv["corr_with_actual"] - dv["corr_with_actual"]) / (dv["corr_with_actual"] + 1e-6)) * 100,
    "dir_improvement": dfv["dir_accuracy"] - dv["dir_accuracy"],
    "REPLACE_CONDITION": {
        "corr_ge_30": dfv["corr_with_actual"] >= 0.30,
        "dir_acc_ge_61": dfv["dir_accuracy"] >= 0.61,
    },
    "VERDICT": "ADOPT DIFFUSION_SELECTOR" if adopt else "KEEP DISTRIBUTION_SELECTOR",
}
with open(output_dir / "diffusion_comparison.json", "w") as f:
    json.dump(summary, f, indent=2)