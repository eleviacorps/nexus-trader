from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from config.project_config import V19_BRANCH_ARCHIVE_PATH, V19_CABR_BASE_MODEL_PATH, V19_CABR_MODEL_PATH, V19_CABR_RL_MODEL_PATH
from src.v13.cabr import CABR, _batch_to_device, augment_cabr_context, load_cabr_model
from scripts.train_cabr_v19 import _resolve_feature_columns


def rl_finetune_cabr(
    *,
    archive_path: Path = V19_BRANCH_ARCHIVE_PATH,
    init_checkpoint_path: Path = V19_CABR_BASE_MODEL_PATH,
    output_path: Path = V19_CABR_RL_MODEL_PATH,
    episodes: int = 50_000,
    lr: float = 1e-5,
    device: str | None = None,
) -> dict[str, object]:
    frame = pd.read_parquet(archive_path)
    frame = augment_cabr_context(frame, mmm_features=None)
    model, branch_cols, context_cols, payload = load_cabr_model(path=init_checkpoint_path, map_location=device or "cpu")
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr))
    grouped = list(frame.sort_values(["sample_id", "branch_id"]).groupby("sample_id"))
    rng = np.random.default_rng(42)
    reward_history: list[float] = []
    for _ in range(max(int(episodes), 1)):
        sample_id, sample = grouped[int(rng.integers(0, len(grouped)))]
        if len(sample) < 2:
            continue
        branch_tensor = torch.as_tensor(sample[list(branch_cols)].fillna(0.0).to_numpy(dtype=np.float32), dtype=torch.float32, device=target_device)
        if getattr(model, "use_temporal", False):
            context_matrix = np.repeat(sample[list(context_cols)].fillna(0.0).to_numpy(dtype=np.float32)[-1:, :], len(sample), axis=0)
            context_tensor = torch.as_tensor(context_matrix[:, None, :].repeat(getattr(model, "n_context_bars", 12), axis=1), dtype=torch.float32, device=target_device)
        else:
            context_tensor = torch.as_tensor(sample[list(context_cols)].fillna(0.0).to_numpy(dtype=np.float32), dtype=torch.float32, device=target_device)
        scores = model.score(branch_tensor, context_tensor)
        distribution = torch.distributions.Categorical(logits=scores)
        index = distribution.sample()
        chosen = sample.iloc[int(index.item())]
        reward = float(chosen.get("actual_final_return", 0.0)) / (1.0 + abs(float(chosen.get("path_error", 0.0))))
        loss = -distribution.log_prob(index) * reward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        reward_history.append(reward)
    rl_payload = dict(payload)
    rl_payload["state_dict"] = model.state_dict()
    rl_payload["rl_reward_mean"] = float(np.mean(reward_history)) if reward_history else 0.0
    rl_payload["rl_reward_history_tail"] = reward_history[-200:]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(rl_payload, output_path)
    return {
        "archive_path": str(archive_path),
        "init_checkpoint_path": str(init_checkpoint_path),
        "output_path": str(output_path),
        "episodes": int(episodes),
        "mean_reward": float(np.mean(reward_history)) if reward_history else 0.0,
        "reward_history_tail": reward_history[-20:],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="RL fine-tune CABR v19 with a simple policy-gradient loop.")
    parser.add_argument("--archive", default=str(V19_BRANCH_ARCHIVE_PATH))
    parser.add_argument("--init", default=str(V19_CABR_BASE_MODEL_PATH))
    parser.add_argument("--episodes", type=int, default=50_000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--output", default=str(V19_CABR_RL_MODEL_PATH))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    report = rl_finetune_cabr(
        archive_path=Path(args.archive),
        init_checkpoint_path=Path(args.init),
        output_path=Path(args.output),
        episodes=int(args.episodes),
        lr=float(args.lr),
        device=args.device,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
