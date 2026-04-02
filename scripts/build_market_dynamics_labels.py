from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover
    pd = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.project_config import (  # noqa: E402
    LEGACY_PRICE_FEATURES_CSV,
    LEGACY_PRICE_FEATURES_PARQUET,
    MARKET_DYNAMICS_LABELS_PATH,
    MARKET_DYNAMICS_REPORT_PATH,
    PRICE_FEATURES_CSV_FALLBACK,
    PRICE_FEATURES_PATH,
    QUANT_FEATURES_CSV_FALLBACK,
    QUANT_FEATURES_PATH,
)
from src.pipeline.fusion import load_price_frame  # noqa: E402
from src.quant.hybrid import merge_quant_features  # noqa: E402
from src.regime.labeling import build_market_dynamics_labels, summarize_market_dynamics  # noqa: E402


def resolve_first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    joined = ", ".join(str(path) for path in paths)
    raise FileNotFoundError(f"No artifact found in: {joined}")


def main() -> int:
    if pd is None:
        raise ImportError("pandas is required to build market-dynamics labels.")

    price_path = resolve_first_existing(
        [PRICE_FEATURES_PATH, PRICE_FEATURES_CSV_FALLBACK, LEGACY_PRICE_FEATURES_PARQUET, LEGACY_PRICE_FEATURES_CSV]
    )
    frame = load_price_frame(price_path)
    quant_path = QUANT_FEATURES_PATH if QUANT_FEATURES_PATH.exists() else QUANT_FEATURES_CSV_FALLBACK if QUANT_FEATURES_CSV_FALLBACK.exists() else None
    if quant_path is not None:
        if quant_path.suffix.lower() == ".parquet":
            quant_frame = pd.read_parquet(quant_path)
        else:
            quant_frame = pd.read_csv(quant_path, index_col=0, parse_dates=True)
        if "timestamp" in quant_frame.columns:
            quant_frame = quant_frame.set_index(pd.to_datetime(quant_frame["timestamp"], errors="coerce")).drop(columns=["timestamp"])
        frame = merge_quant_features(frame, quant_frame)

    labels = build_market_dynamics_labels(frame)
    output = pd.concat([frame[["open", "high", "low", "close"]], labels], axis=1)
    MARKET_DYNAMICS_LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    saved_path = MARKET_DYNAMICS_LABELS_PATH
    try:
        output.to_parquet(MARKET_DYNAMICS_LABELS_PATH)
    except Exception:
        saved_path = MARKET_DYNAMICS_LABELS_PATH.with_suffix(".csv")
        output.to_csv(saved_path)

    report = summarize_market_dynamics(labels).to_dict()
    report["artifact_path"] = str(saved_path)
    MARKET_DYNAMICS_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    MARKET_DYNAMICS_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
