from __future__ import annotations

import os
from pathlib import Path

LOCAL_PROJECT_ROOT = Path(r"C:/PersonalDrive/Programming/AiStudio/nexus-trader")
REMOTE_DATA_ROOT = Path("/home/rocm-user/jupyter/nexus")

RUNNING_ON_SERVER = REMOTE_DATA_ROOT.exists()
PROJECT_ROOT = REMOTE_DATA_ROOT if RUNNING_ON_SERVER else LOCAL_PROJECT_ROOT
USE_REMOTE_DATA = RUNNING_ON_SERVER or os.getenv("NEXUS_USE_REMOTE_DATA", "0") == "1"
DATA_ROOT = REMOTE_DATA_ROOT if USE_REMOTE_DATA else LOCAL_PROJECT_ROOT

SEQUENCE_LEN = 120
LOOKAHEAD = 5
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

FEATURE_DIM_PRICE = 36
FEATURE_DIM_NEWS = 32
FEATURE_DIM_CROWD = 32
FEATURE_DIM_TOTAL = FEATURE_DIM_PRICE + FEATURE_DIM_NEWS + FEATURE_DIM_CROWD

FILL_LIMIT = 60

BATCH_SIZE_SERVER = 2048
BATCH_SIZE_LOCAL = 1024
NUM_WORKERS = 0

TRAIN_YEARS = tuple(range(2009, 2021))
VAL_YEARS = (2021, 2022, 2023)
TEST_YEARS = (2024, 2025, 2026)

NOTEBOOK_PIPELINE = [
    "00_environment_setup.ipynb",
    "01_data_download.ipynb",
    "02_price_pipeline.ipynb",
    "03_news_pipeline.ipynb",
    "04_crowd_pipeline.ipynb",
    "05_persona_simulation.ipynb",
    "06_feature_fusion.ipynb",
    "07_tft_training.ipynb",
    "08_future_branching.ipynb",
    "09_reverse_collapse_and_ui.ipynb",
    "10_validation_and_tests.ipynb",
]

PRICE_FEATURE_COLUMNS = [
    "return_1",
    "return_3",
    "return_6",
    "return_12",
    "rsi_14",
    "rsi_7",
    "macd_hist",
    "macd",
    "macd_sig",
    "stoch_k",
    "stoch_d",
    "ema_9_ratio",
    "ema_21_ratio",
    "ema_50_ratio",
    "ema_cross",
    "atr_pct",
    "bb_width",
    "bb_pct",
    "body_pct",
    "upper_wick",
    "lower_wick",
    "is_bullish",
    "displacement",
    "dist_to_high",
    "dist_to_low",
    "hh",
    "ll",
    "volume_ratio",
    "session_asian",
    "session_london",
    "session_ny",
    "session_overlap",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_PRICE_DIR = RAW_DATA_DIR / "price"
RAW_NEWS_DIR = RAW_DATA_DIR / "news"
RAW_CROWD_DIR = RAW_DATA_DIR / "crowd"
RAW_MACRO_DIR = RAW_DATA_DIR / "macro"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
FEATURES_DIR = DATA_DIR / "features"
BRANCHES_DIR = DATA_DIR / "branches"

MODELS_DIR = PROJECT_ROOT / "models"
NEWS_PROJECTION_DIR = MODELS_DIR / "news_projection"
CROWD_PROJECTION_DIR = MODELS_DIR / "crowd_projection"
PERSONA_MODEL_DIR = MODELS_DIR / "personas"
TFT_MODEL_DIR = MODELS_DIR / "tft"
COLLAPSE_MODEL_DIR = MODELS_DIR / "collapse"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_CHARTS_DIR = OUTPUTS_DIR / "charts"
OUTPUTS_CONES_DIR = OUTPUTS_DIR / "probability_cones"
OUTPUTS_EVAL_DIR = OUTPUTS_DIR / "evaluation"
OUTPUTS_LOGS_DIR = OUTPUTS_DIR / "logs"

LEGACY_DATA_STORE_DIR = PROJECT_ROOT / "data_store"
LEGACY_PROCESSED_DIR = LEGACY_DATA_STORE_DIR / "processed"
LEGACY_EMBEDDINGS_DIR = LEGACY_DATA_STORE_DIR / "embeddings"
LEGACY_SYNTHETIC_DIR = LEGACY_DATA_STORE_DIR / "synthetic"
LEGACY_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LEGACY_TFT_CHECKPOINT_PATH = LEGACY_CHECKPOINT_DIR / "tft" / "tft_best.pt"

CHECKPOINT_DIR = MODELS_DIR
TFT_CHECKPOINT_DIR = TFT_MODEL_DIR
TFT_CHECKPOINT_PATH = TFT_MODEL_DIR / "final_tft.ckpt"
NEWS_HEAD_CHECKPOINT_PATH = NEWS_PROJECTION_DIR / "news_head_supervised.pt"
CROWD_HEAD_CHECKPOINT_PATH = CROWD_PROJECTION_DIR / "crowd_head_supervised.pt"

PRICE_FEATURES_PATH = FEATURES_DIR / "price_features.parquet"
PRICE_FEATURES_CSV_FALLBACK = FEATURES_DIR / "price_features.csv"
LEGACY_PRICE_FEATURES_CSV = LEGACY_PROCESSED_DIR / "XAUUSD_1m_features.csv"
LEGACY_PRICE_FEATURES_PARQUET = LEGACY_PROCESSED_DIR / "XAUUSD_1m_features.parquet"

NEWS_EMBEDDINGS_PATH = EMBEDDINGS_DIR / "news_embedding.parquet"
CROWD_EMBEDDINGS_PATH = EMBEDDINGS_DIR / "crowd_embedding.parquet"
NEWS_EMBEDDINGS_INDEX_PATH = EMBEDDINGS_DIR / "news_emb_index.parquet"
CROWD_EMBEDDINGS_INDEX_PATH = EMBEDDINGS_DIR / "crowd_emb_index.parquet"
NEWS_EMBEDDINGS_NPY_PATH = EMBEDDINGS_DIR / "news_embeddings_32.npy"
NEWS_EMBEDDINGS_RAW_PATH = EMBEDDINGS_DIR / "news_embeddings.npy"
CROWD_EMBEDDINGS_NPY_PATH = EMBEDDINGS_DIR / "crowd_embeddings.npy"
LEGACY_NEWS_EMBEDDINGS_NPY_PATH = LEGACY_EMBEDDINGS_DIR / "news_embeddings_32.npy"
LEGACY_NEWS_EMBEDDINGS_RAW_PATH = LEGACY_EMBEDDINGS_DIR / "news_embeddings.npy"
LEGACY_CROWD_EMBEDDINGS_NPY_PATH = LEGACY_EMBEDDINGS_DIR / "crowd_embeddings.npy"
LEGACY_NEWS_EMBEDDINGS_INDEX_PATH = LEGACY_EMBEDDINGS_DIR / "news_emb_index.parquet"
LEGACY_CROWD_EMBEDDINGS_INDEX_PATH = LEGACY_EMBEDDINGS_DIR / "crowd_emb_index.parquet"

PERSONA_OUTPUTS_PATH = PROCESSED_DATA_DIR / "persona_outputs.parquet"
PERSONA_WEIGHT_HISTORY_PATH = PROCESSED_DATA_DIR / "persona_weight_history.parquet"
FUSED_TENSOR_PATH = FEATURES_DIR / "fused_tensor.npy"
TARGETS_PATH = FEATURES_DIR / "targets.npy"
FUSED_FEATURE_MATRIX_PATH = FEATURES_DIR / "fused_features.npy"
FUSED_TIMESTAMPS_PATH = FEATURES_DIR / "timestamps.npy"
FUSION_REPORT_PATH = OUTPUTS_EVAL_DIR / "fusion_report.json"
FEATURE_IMPORTANCE_REPORT_PATH = OUTPUTS_EVAL_DIR / "feature_importance.json"
CALIBRATION_REPORT_PATH = OUTPUTS_EVAL_DIR / "calibration_report.json"
TRAINING_SUMMARY_PATH = OUTPUTS_EVAL_DIR / "training_summary.json"
FUTURE_BRANCHES_PATH = BRANCHES_DIR / "future_branches.json"
FINAL_TFT_METRICS_PATH = OUTPUTS_EVAL_DIR / "tft_metrics.json"
MODEL_MANIFEST_PATH = TFT_MODEL_DIR / "model_manifest.json"
PROBABILITY_CONE_HTML_PATH = OUTPUTS_CHARTS_DIR / "probability_cone.html"
PERSONA_BREAKDOWN_HTML_PATH = OUTPUTS_CHARTS_DIR / "persona_breakdown.html"
MODEL_SERVICE_HOST = os.getenv("NEXUS_MODEL_HOST", "0.0.0.0")
MODEL_SERVICE_PORT = int(os.getenv("NEXUS_MODEL_PORT", "8000"))

NORM_STATS_PATH = PROJECT_ROOT / "config" / "norm_stats_1m.json"
PERSONA_CONFIG_PATH = PROJECT_ROOT / "config" / "persona_config.json"


def get_data_path(*parts: str) -> Path:
    return DATA_DIR.joinpath(*parts)


def get_local_path(*parts: str) -> Path:
    return LOCAL_PROJECT_ROOT.joinpath(*parts)


def get_project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)
