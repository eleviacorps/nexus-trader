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

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
TFT_CHECKPOINT_DIR = CHECKPOINT_DIR / "tft"
TFT_CHECKPOINT_PATH = TFT_CHECKPOINT_DIR / "tft_best.pt"
NEWS_HEAD_CHECKPOINT_PATH = CHECKPOINT_DIR / "news_head_supervised.pt"
CROWD_HEAD_CHECKPOINT_PATH = CHECKPOINT_DIR / "crowd_head_supervised.pt"

DATA_STORE_DIR = DATA_ROOT / "data_store"
PROCESSED_DIR = DATA_STORE_DIR / "processed"
EMBEDDINGS_DIR = DATA_STORE_DIR / "embeddings"
SYNTHETIC_DIR = DATA_STORE_DIR / "synthetic"

PRICE_FEATURES_PATH = PROCESSED_DIR / "XAUUSD_1m_features.csv"
PRICE_FEATURES_PARQUET_PATH = PROCESSED_DIR / "XAUUSD_1m_features.parquet"
NEWS_EMBEDDINGS_PATH = EMBEDDINGS_DIR / "news_embeddings_32.npy"
NEWS_EMBEDDINGS_RAW_PATH = EMBEDDINGS_DIR / "news_embeddings.npy"
NEWS_EMBEDDINGS_INDEX_PATH = EMBEDDINGS_DIR / "news_emb_index.parquet"
CROWD_EMBEDDINGS_PATH = EMBEDDINGS_DIR / "crowd_embeddings.npy"
CROWD_EMBEDDINGS_INDEX_PATH = EMBEDDINGS_DIR / "crowd_emb_index.parquet"

NORM_STATS_PATH = PROJECT_ROOT / "config" / "norm_stats_1m.json"
PERSONA_CONFIG_PATH = PROJECT_ROOT / "config" / "persona_config.json"


def get_data_path(*parts: str) -> Path:
    return DATA_ROOT.joinpath(*parts)


def get_local_path(*parts: str) -> Path:
    return LOCAL_PROJECT_ROOT.joinpath(*parts)


def get_project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)
