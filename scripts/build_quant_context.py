from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.project_config import QUANT_FEATURES_CSV_FALLBACK, QUANT_FEATURES_PATH, QUANT_REPORT_PATH  # noqa: E402
from src.pipeline.fusion import load_price_frame  # noqa: E402
from src.quant.hybrid import build_quant_features, summarize_quant_frame  # noqa: E402


def _resolve_price_path() -> Path:
    from config.project_config import (  # noqa: E402
        LEGACY_PRICE_FEATURES_CSV,
        LEGACY_PRICE_FEATURES_PARQUET,
        PRICE_FEATURES_CSV_FALLBACK,
        PRICE_FEATURES_PATH,
    )

    for path in [PRICE_FEATURES_PATH, PRICE_FEATURES_CSV_FALLBACK, LEGACY_PRICE_FEATURES_PARQUET, LEGACY_PRICE_FEATURES_CSV]:
        if path.exists():
            return path
    raise FileNotFoundError("Price features are required before building quant context.")


def main() -> int:
    price_path = _resolve_price_path()
    frame = load_price_frame(price_path)
    quant = build_quant_features(frame)
    QUANT_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        quant.to_parquet(QUANT_FEATURES_PATH)
        quant_path = QUANT_FEATURES_PATH
    except Exception:
        quant.to_csv(QUANT_FEATURES_CSV_FALLBACK)
        quant_path = QUANT_FEATURES_CSV_FALLBACK
    report = summarize_quant_frame(quant)
    payload = report.__dict__ | {"quant_path": str(quant_path)}
    QUANT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUANT_REPORT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
