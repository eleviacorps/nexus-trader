from __future__ import annotations

from pathlib import Path
from typing import List

from config.project_config import (
    BRANCHES_DIR,
    CROWD_PROJECTION_DIR,
    DATA_DIR,
    FEATURES_DIR,
    MODELS_DIR,
    NOTEBOOKS_DIR,
    OUTPUTS_CHARTS_DIR,
    OUTPUTS_CONES_DIR,
    OUTPUTS_DIR,
    OUTPUTS_EVAL_DIR,
    OUTPUTS_LOGS_DIR,
    PERSONA_MODEL_DIR,
    PROCESSED_DATA_DIR,
    RAW_CROWD_DIR,
    RAW_MACRO_DIR,
    RAW_NEWS_DIR,
    RAW_PRICE_DIR,
    TFT_MODEL_DIR,
    NEWS_PROJECTION_DIR,
    PROJECT_ROOT,
)


CLOUD_TREE: List[Path] = [
    NOTEBOOKS_DIR,
    RAW_PRICE_DIR,
    RAW_NEWS_DIR,
    RAW_CROWD_DIR,
    RAW_MACRO_DIR,
    PROCESSED_DATA_DIR,
    DATA_DIR / "embeddings",
    FEATURES_DIR,
    BRANCHES_DIR,
    NEWS_PROJECTION_DIR,
    CROWD_PROJECTION_DIR,
    PERSONA_MODEL_DIR,
    TFT_MODEL_DIR,
    MODELS_DIR / "collapse",
    OUTPUTS_DIR,
    OUTPUTS_CHARTS_DIR,
    OUTPUTS_CONES_DIR,
    OUTPUTS_EVAL_DIR,
    OUTPUTS_LOGS_DIR,
    PROJECT_ROOT / "tests",
    PROJECT_ROOT / "src",
]


def create_project_tree() -> List[Path]:
    created: List[Path] = []
    for path in CLOUD_TREE:
        path.mkdir(parents=True, exist_ok=True)
        created.append(path)
    return created
