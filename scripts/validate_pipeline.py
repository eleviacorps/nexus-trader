from __future__ import annotations

import ast
import csv
import importlib.util
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

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


def resolve_first_existing(paths: Sequence[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def torch_forward_check() -> CheckResult:
    spec = importlib.util.find_spec("torch")
    if spec is None:
        return CheckResult("torch_forward_pass", "BLOCKED", "torch is not installed locally")
    return CheckResult("torch_forward_pass", "WARN", "torch is available but forward validation is deferred to the server runtime")


def main() -> int:
    from config.project_config import (
        CROWD_EMBEDDINGS_INDEX_PATH,
        CROWD_EMBEDDINGS_NPY_PATH,
        FEATURE_DIM_CROWD,
        FEATURE_DIM_NEWS,
        FEATURE_DIM_TOTAL,
        CROWD_EVENTS_PATH,
        LEGACY_CROWD_EMBEDDINGS_INDEX_PATH,
        LEGACY_CROWD_EMBEDDINGS_NPY_PATH,
        LEGACY_NEWS_EMBEDDINGS_INDEX_PATH,
        LEGACY_NEWS_EMBEDDINGS_NPY_PATH,
        LEGACY_NEWS_EMBEDDINGS_RAW_PATH,
        LEGACY_PRICE_FEATURES_CSV,
        LEGACY_PRICE_FEATURES_PARQUET,
        LEGACY_TFT_CHECKPOINT_PATH,
        MACRO_FEATURES_PATH,
        NEWS_EVENTS_PATH,
        NEWS_EMBEDDINGS_INDEX_PATH,
        NEWS_EMBEDDINGS_NPY_PATH,
        NEWS_EMBEDDINGS_RAW_PATH,
        FUSED_FEATURE_MATRIX_PATH,
        PERSONA_OUTPUTS_PATH,
        PERSONA_WEIGHT_HISTORY_PATH,
        PRICE_FEATURES_CSV_FALLBACK,
        PRICE_FEATURES_PATH,
        PRICE_FEATURE_COLUMNS,
        SIM_CONFIDENCE_PATH,
        SIM_TARGETS_PATH,
        TARGETS_MULTIHORIZON_PATH,
        TFT_CHECKPOINT_PATH,
    )

    results: List[CheckResult] = []

    price_path = resolve_first_existing([PRICE_FEATURES_PATH, PRICE_FEATURES_CSV_FALLBACK, LEGACY_PRICE_FEATURES_CSV, LEGACY_PRICE_FEATURES_PARQUET])
    news_path = resolve_first_existing([NEWS_EMBEDDINGS_RAW_PATH, NEWS_EMBEDDINGS_NPY_PATH, LEGACY_NEWS_EMBEDDINGS_RAW_PATH, LEGACY_NEWS_EMBEDDINGS_NPY_PATH])
    crowd_path = resolve_first_existing([CROWD_EMBEDDINGS_NPY_PATH, LEGACY_CROWD_EMBEDDINGS_NPY_PATH])
    news_index_path = resolve_first_existing([NEWS_EMBEDDINGS_INDEX_PATH, LEGACY_NEWS_EMBEDDINGS_INDEX_PATH])
    crowd_index_path = resolve_first_existing([CROWD_EMBEDDINGS_INDEX_PATH, LEGACY_CROWD_EMBEDDINGS_INDEX_PATH])
    checkpoint_path = resolve_first_existing([TFT_CHECKPOINT_PATH, LEGACY_TFT_CHECKPOINT_PATH])

    if price_path and price_path.suffix.lower() == ".csv":
        header = read_csv_header(price_path)
        missing = [column for column in PRICE_FEATURE_COLUMNS if column not in header]
        if missing:
            results.append(CheckResult("price_features", "FAIL", f"missing columns: {', '.join(missing)}"))
        else:
            results.append(CheckResult("price_features", "PASS", f"{len(PRICE_FEATURE_COLUMNS)} configured columns found via {price_path.name}"))
    elif price_path:
        results.append(CheckResult("price_features", "PASS", f"found parquet artifact {price_path.name}"))
    else:
        results.append(CheckResult("price_features", "FAIL", "no price feature artifact found"))

    if news_path:
        shape = read_npy_shape(news_path)
        status = "PASS" if len(shape) == 2 and shape[1] == FEATURE_DIM_NEWS else "FAIL"
        results.append(CheckResult("news_embeddings", status, f"shape={shape} source={news_path.name}"))
    else:
        results.append(CheckResult("news_embeddings", "FAIL", "no news embedding tensor found"))

    if crowd_path:
        shape = read_npy_shape(crowd_path)
        status = "PASS" if len(shape) == 2 and shape[1] == FEATURE_DIM_CROWD else "FAIL"
        results.append(CheckResult("crowd_embeddings", status, f"shape={shape} source={crowd_path.name}"))
    else:
        results.append(CheckResult("crowd_embeddings", "FAIL", "no crowd embedding tensor found"))

    results.append(CheckResult(
        "embedding_indexes",
        "PASS" if news_index_path and crowd_index_path else "FAIL",
        f"news_index={bool(news_index_path)} crowd_index={bool(crowd_index_path)}",
    ))

    results.append(CheckResult(
        "macro_features",
        "PASS" if MACRO_FEATURES_PATH.exists() else "WARN",
        str(MACRO_FEATURES_PATH) if MACRO_FEATURES_PATH.exists() else "macro context not built yet",
    ))
    results.append(CheckResult(
        "news_events",
        "PASS" if NEWS_EVENTS_PATH.exists() else "WARN",
        str(NEWS_EVENTS_PATH) if NEWS_EVENTS_PATH.exists() else "news events parquet not built yet",
    ))
    results.append(CheckResult(
        "crowd_events",
        "PASS" if CROWD_EVENTS_PATH.exists() else "WARN",
        str(CROWD_EVENTS_PATH) if CROWD_EVENTS_PATH.exists() else "crowd events parquet not built yet",
    ))
    results.append(CheckResult(
        "persona_outputs",
        "PASS" if PERSONA_OUTPUTS_PATH.exists() else "WARN",
        str(PERSONA_OUTPUTS_PATH) if PERSONA_OUTPUTS_PATH.exists() else "persona outputs parquet not built yet",
    ))
    results.append(CheckResult(
        "persona_weight_history",
        "PASS" if PERSONA_WEIGHT_HISTORY_PATH.exists() else "WARN",
        str(PERSONA_WEIGHT_HISTORY_PATH) if PERSONA_WEIGHT_HISTORY_PATH.exists() else "persona weight history parquet not built yet",
    ))
    results.append(CheckResult(
        "simulation_targets",
        "PASS" if SIM_TARGETS_PATH.exists() and SIM_CONFIDENCE_PATH.exists() else "WARN",
        f"sim_targets={SIM_TARGETS_PATH.exists()} sim_confidence={SIM_CONFIDENCE_PATH.exists()}",
    ))
    results.append(CheckResult(
        "multihorizon_targets",
        "PASS" if TARGETS_MULTIHORIZON_PATH.exists() else "WARN",
        str(TARGETS_MULTIHORIZON_PATH) if TARGETS_MULTIHORIZON_PATH.exists() else "multi-horizon target bundle not built yet",
    ))

    fused_path = FUSED_FEATURE_MATRIX_PATH if FUSED_FEATURE_MATRIX_PATH.exists() else None
    if fused_path is not None:
        fused_rows, fused_width = read_npy_shape(fused_path)
        results.append(CheckResult("row_alignment", "PASS", f"fused_rows={fused_rows} final_width={fused_width}"))
    elif price_path and price_path.suffix.lower() == ".csv" and news_path and crowd_path:
        price_rows = count_csv_rows(price_path)
        news_rows = read_npy_shape(news_path)[0]
        crowd_rows = read_npy_shape(crowd_path)[0]
        if price_rows == news_rows == crowd_rows:
            results.append(CheckResult("row_alignment", "PASS", f"row_count={price_rows} final_width={FEATURE_DIM_TOTAL}"))
        else:
            results.append(CheckResult("row_alignment", "FAIL", f"price={price_rows} news={news_rows} crowd={crowd_rows}"))
    else:
        results.append(CheckResult("row_alignment", "WARN", "row alignment check deferred until compatible row-count artifacts exist"))

    results.append(CheckResult(
        "checkpoint",
        "PASS" if checkpoint_path else "FAIL",
        str(checkpoint_path) if checkpoint_path else "no checkpoint found",
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
