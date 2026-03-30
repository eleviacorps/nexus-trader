from __future__ import annotations

import ast
import csv
import importlib.util
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str


def read_csv_header(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        return next(reader)


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def read_npy_shape(path: Path):
    with path.open("rb") as handle:
        magic = handle.read(6)
        if magic != b"\x93NUMPY":
            raise ValueError(f"{path.name} is not a valid .npy file")
        major, minor = struct.unpack("BB", handle.read(2))
        header_len_size = 2 if major == 1 else 4
        header_len = struct.unpack("<H" if header_len_size == 2 else "<I", handle.read(header_len_size))[0]
        header = handle.read(header_len).decode("latin1")
        metadata = ast.literal_eval(header)
        return metadata["shape"]


def torch_forward_check() -> CheckResult:
    spec = importlib.util.find_spec("torch")
    if spec is None:
        return CheckResult("torch_forward_pass", "BLOCKED", "torch is not installed locally")
    return CheckResult("torch_forward_pass", "WARN", "torch is available but forward validation is deferred to the server runtime")


def main() -> int:
    from config.project_config import (
        CROWD_EMBEDDINGS_INDEX_PATH,
        CROWD_EMBEDDINGS_PATH,
        FEATURE_DIM_CROWD,
        FEATURE_DIM_NEWS,
        FEATURE_DIM_TOTAL,
        NEWS_EMBEDDINGS_INDEX_PATH,
        NEWS_EMBEDDINGS_PATH,
        PRICE_FEATURES_PATH,
        PRICE_FEATURE_COLUMNS,
        TFT_CHECKPOINT_PATH,
    )

    results: List[CheckResult] = []

    if PRICE_FEATURES_PATH.exists():
        header = read_csv_header(PRICE_FEATURES_PATH)
        missing = [column for column in PRICE_FEATURE_COLUMNS if column not in header]
        if missing:
            results.append(CheckResult("price_features", "FAIL", f"missing columns: {', '.join(missing)}"))
        else:
            results.append(CheckResult("price_features", "PASS", f"{len(PRICE_FEATURE_COLUMNS)} configured columns found"))
    else:
        results.append(CheckResult("price_features", "FAIL", f"missing file: {PRICE_FEATURES_PATH}"))

    if NEWS_EMBEDDINGS_PATH.exists():
        shape = read_npy_shape(NEWS_EMBEDDINGS_PATH)
        status = "PASS" if len(shape) == 2 and shape[1] == FEATURE_DIM_NEWS else "FAIL"
        results.append(CheckResult("news_embeddings", status, f"shape={shape}"))
    else:
        results.append(CheckResult("news_embeddings", "FAIL", f"missing file: {NEWS_EMBEDDINGS_PATH}"))

    if CROWD_EMBEDDINGS_PATH.exists():
        shape = read_npy_shape(CROWD_EMBEDDINGS_PATH)
        status = "PASS" if len(shape) == 2 and shape[1] == FEATURE_DIM_CROWD else "FAIL"
        results.append(CheckResult("crowd_embeddings", status, f"shape={shape}"))
    else:
        results.append(CheckResult("crowd_embeddings", "FAIL", f"missing file: {CROWD_EMBEDDINGS_PATH}"))

    results.append(CheckResult(
        "embedding_indexes",
        "PASS" if NEWS_EMBEDDINGS_INDEX_PATH.exists() and CROWD_EMBEDDINGS_INDEX_PATH.exists() else "FAIL",
        f"news_index={NEWS_EMBEDDINGS_INDEX_PATH.exists()} crowd_index={CROWD_EMBEDDINGS_INDEX_PATH.exists()}",
    ))

    if PRICE_FEATURES_PATH.exists() and NEWS_EMBEDDINGS_PATH.exists() and CROWD_EMBEDDINGS_PATH.exists():
        price_rows = count_csv_rows(PRICE_FEATURES_PATH)
        news_rows = read_npy_shape(NEWS_EMBEDDINGS_PATH)[0]
        crowd_rows = read_npy_shape(CROWD_EMBEDDINGS_PATH)[0]
        if price_rows == news_rows == crowd_rows:
            results.append(CheckResult("row_alignment", "PASS", f"row_count={price_rows} final_width={FEATURE_DIM_TOTAL}"))
        else:
            results.append(CheckResult("row_alignment", "FAIL", f"price={price_rows} news={news_rows} crowd={crowd_rows}"))
    else:
        results.append(CheckResult("row_alignment", "FAIL", "cannot compare row counts until all feature blocks exist"))

    results.append(CheckResult(
        "checkpoint",
        "PASS" if TFT_CHECKPOINT_PATH.exists() else "FAIL",
        str(TFT_CHECKPOINT_PATH),
    ))

    results.append(torch_forward_check())

    print("Validation Summary")
    print("=" * 18)
    for result in results:
        print(f"[{result.status}] {result.name}: {result.detail}")

    failed = any(result.status == "FAIL" for result in results)
    print("PASS" if not failed else "FAIL")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
