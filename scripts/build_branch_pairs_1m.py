from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.project_config import V19_BRANCH_ARCHIVE_PATH, V20_BRANCH_PAIR_DATASET_PATH


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the V20 branch-pair training dataset from the V19 archive.")
    parser.add_argument("--max-pairs", type=int, default=100000)
    args = parser.parse_args()

    archive = pd.read_parquet(V19_BRANCH_ARCHIVE_PATH)
    rows: list[dict[str, object]] = []
    for _, group in archive.groupby("sample_id"):
        ordered = group.sort_values("winning_branch", ascending=False).head(8)
        if len(ordered) < 2:
            continue
        top = ordered.iloc[0]
        for _, other in ordered.iloc[1:].iterrows():
            rows.append(
                {
                    "sample_id": int(top["sample_id"]),
                    "branch_a": int(top["branch_id"]),
                    "branch_b": int(other["branch_id"]),
                    "a_win": 1,
                    "a_direction": float(top.get("branch_direction", 0.0)),
                    "b_direction": float(other.get("branch_direction", 0.0)),
                    "a_similarity": float(top.get("analog_similarity", 0.5)),
                    "b_similarity": float(other.get("analog_similarity", 0.5)),
                    "regime": str(top.get("dominant_regime", "range")),
                }
            )
            if len(rows) >= args.max_pairs:
                break
        if len(rows) >= args.max_pairs:
            break
    frame = pd.DataFrame(rows)
    V20_BRANCH_PAIR_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(V20_BRANCH_PAIR_DATASET_PATH)
    print(f"saved={V20_BRANCH_PAIR_DATASET_PATH}")
    print(f"rows={len(frame)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
