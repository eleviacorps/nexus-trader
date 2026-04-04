from __future__ import annotations

import os
from pathlib import Path

LOCAL_PROJECT_ROOT = Path(r"C:/PersonalDrive/Programming/AiStudio/nexus-trader")
REMOTE_DATA_ROOT = Path("/home/rocm-user/jupyter/nexus")

RUNNING_ON_SERVER = REMOTE_DATA_ROOT.exists()
PROJECT_ROOT = REMOTE_DATA_ROOT if RUNNING_ON_SERVER else LOCAL_PROJECT_ROOT
USE_REMOTE_DATA = RUNNING_ON_SERVER or os.getenv("NEXUS_USE_REMOTE_DATA", "0") == "1"
DATA_ROOT = REMOTE_DATA_ROOT if USE_REMOTE_DATA else LOCAL_PROJECT_ROOT

SEQUENCE_LEN = int(os.getenv("NEXUS_SEQUENCE_LEN", "120"))
LOOKAHEAD = int(os.getenv("NEXUS_LOOKAHEAD", "5"))
TRAIN_SPLIT = float(os.getenv("NEXUS_TRAIN_SPLIT", "0.70"))
VAL_SPLIT = float(os.getenv("NEXUS_VAL_SPLIT", "0.15"))
TEST_SPLIT = float(os.getenv("NEXUS_TEST_SPLIT", "0.15"))

FEATURE_DIM_PRICE = 36
FEATURE_DIM_NEWS = 32
FEATURE_DIM_CROWD = 32
FEATURE_DIM_TOTAL = FEATURE_DIM_PRICE + FEATURE_DIM_NEWS + FEATURE_DIM_CROWD

FILL_LIMIT = 60

BATCH_SIZE_SERVER = 2048
BATCH_SIZE_LOCAL = 1024
NUM_WORKERS_SERVER = int(os.getenv("NEXUS_NUM_WORKERS_SERVER", "8"))
NUM_WORKERS_LOCAL = int(os.getenv("NEXUS_NUM_WORKERS_LOCAL", "2"))
NUM_WORKERS = NUM_WORKERS_SERVER if RUNNING_ON_SERVER else NUM_WORKERS_LOCAL
PREFETCH_FACTOR = int(os.getenv("NEXUS_PREFETCH_FACTOR", "4"))
PERSISTENT_WORKERS = os.getenv("NEXUS_PERSISTENT_WORKERS", "1") == "1"
PIN_MEMORY = os.getenv("NEXUS_PIN_MEMORY", "1") == "1"
AMP_ENABLED = os.getenv("NEXUS_AMP_ENABLED", "1") == "1"
AMP_DTYPE = os.getenv("NEXUS_AMP_DTYPE", "bfloat16")
TORCH_COMPILE_ENABLED = os.getenv("NEXUS_TORCH_COMPILE", "0") == "1"

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
OUTPUTS_V8_DIR = OUTPUTS_DIR / "v8"
OUTPUTS_V9_DIR = OUTPUTS_DIR / "v9"
OUTPUTS_V10_DIR = OUTPUTS_DIR / "v10"
OUTPUTS_V11_DIR = OUTPUTS_DIR / "v11"

LEGACY_DATA_STORE_DIR = PROJECT_ROOT / "data_store"
LEGACY_PROCESSED_DIR = LEGACY_DATA_STORE_DIR / "processed"
LEGACY_EMBEDDINGS_DIR = LEGACY_DATA_STORE_DIR / "embeddings"
LEGACY_SYNTHETIC_DIR = LEGACY_DATA_STORE_DIR / "synthetic"
LEGACY_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
V9_CHECKPOINT_DIR = LEGACY_CHECKPOINT_DIR / "v9"
V10_CHECKPOINT_DIR = LEGACY_CHECKPOINT_DIR / "v10"
V11_CHECKPOINT_DIR = LEGACY_CHECKPOINT_DIR / "v11"
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
MACRO_FEATURES_PATH = PROCESSED_DATA_DIR / "macro_features.parquet"
QUANT_FEATURES_PATH = PROCESSED_DATA_DIR / "quant_features.parquet"
QUANT_FEATURES_CSV_FALLBACK = PROCESSED_DATA_DIR / "quant_features.csv"
MARKET_DYNAMICS_LABELS_PATH = PROCESSED_DATA_DIR / "market_dynamics_labels.parquet"
NEWS_EVENTS_PATH = PROCESSED_DATA_DIR / "news_events.parquet"
CROWD_EVENTS_PATH = PROCESSED_DATA_DIR / "crowd_events.parquet"
SIM_TARGETS_PATH = FEATURES_DIR / "sim_targets.npy"
SIM_CONFIDENCE_PATH = FEATURES_DIR / "sim_confidence.npy"
SAMPLE_WEIGHTS_PATH = FEATURES_DIR / "sample_weights.npy"
FUSED_TENSOR_PATH = FEATURES_DIR / "fused_tensor.npy"
TARGETS_PATH = FEATURES_DIR / "targets.npy"
TARGETS_MULTIHORIZON_PATH = FEATURES_DIR / "targets_multihorizon.npz"
TARGET_HOLD_MASK_PATH = FEATURES_DIR / "target_hold_mask.npy"
GATE_CONTEXT_PATH = FEATURES_DIR / "gate_context.npy"
FUSED_FEATURE_MATRIX_PATH = FEATURES_DIR / "fused_features.npy"
FUSED_TIMESTAMPS_PATH = FEATURES_DIR / "timestamps.npy"
FUSION_REPORT_PATH = OUTPUTS_EVAL_DIR / "fusion_report.json"
FEATURE_IMPORTANCE_REPORT_PATH = OUTPUTS_EVAL_DIR / "feature_importance.json"
CALIBRATION_REPORT_PATH = OUTPUTS_EVAL_DIR / "calibration_report.json"
TRAINING_SUMMARY_PATH = OUTPUTS_EVAL_DIR / "training_summary.json"
FUTURE_BRANCHES_PATH = BRANCHES_DIR / "future_branches.json"
LATEST_MARKET_SNAPSHOT_PATH = OUTPUTS_EVAL_DIR / "latest_market_snapshot.json"
LIVE_SIMULATION_HISTORY_PATH = OUTPUTS_EVAL_DIR / "live_simulation_history.json"
FINAL_TFT_METRICS_PATH = OUTPUTS_EVAL_DIR / "tft_metrics.json"
ANALOG_REPORT_PATH = OUTPUTS_EVAL_DIR / "analog_report.json"
WALKFORWARD_REPORT_PATH = OUTPUTS_EVAL_DIR / "walkforward_report.json"
BACKTEST_REPORT_PATH = OUTPUTS_EVAL_DIR / "backtest_report.json"
MODEL_MANIFEST_PATH = TFT_MODEL_DIR / "model_manifest.json"
PRECISION_GATE_PATH = TFT_MODEL_DIR / "precision_gate.json"
META_GATE_PATH = TFT_MODEL_DIR / "meta_gate.pkl"
V8_HMM_FRAME_PATH = OUTPUTS_V8_DIR / "hmm_regime_features.parquet"
V8_HMM_MODEL_PATH = OUTPUTS_V8_DIR / "hmm_regime.pkl"
V8_GARCH_FRAME_PATH = OUTPUTS_V8_DIR / "garch_volatility_features.parquet"
V8_GARCH_MODEL_PATH = OUTPUTS_V8_DIR / "garch_model.pkl"
V8_FAIR_VALUE_FRAME_PATH = OUTPUTS_V8_DIR / "fair_value_features.parquet"
V8_FAIR_VALUE_MODEL_PATH = OUTPUTS_V8_DIR / "fair_value_model.pkl"
V8_ANALOG_CACHE_PATH = OUTPUTS_V8_DIR / "analog_cache.npz"
V8_ANALOG_CACHE_META_PATH = OUTPUTS_V8_DIR / "analog_cache_meta.json"
V8_BRANCH_ARCHIVE_PATH = OUTPUTS_V8_DIR / "branch_archive.parquet"
V8_BRANCH_SELECTOR_PATH = OUTPUTS_V8_DIR / "branch_selector.pkl"
V8_QUANT_STACK_REPORT_PATH = OUTPUTS_EVAL_DIR / "v8_quant_stack_report.json"
V8_BRANCH_ARCHIVE_REPORT_PATH = OUTPUTS_EVAL_DIR / "v8_branch_archive_report.json"
V8_BRANCH_SELECTOR_REPORT_PATH = OUTPUTS_EVAL_DIR / "v8_branch_selector_report.json"
V8_EVALUATION_REPORT_PATH = OUTPUTS_EVAL_DIR / "v8_evaluation_report.json"
V8_SUMMARY_JSON_PATH = OUTPUTS_EVAL_DIR / "v8_summary.json"
V8_SUMMARY_MD_PATH = OUTPUTS_EVAL_DIR / "v8_summary.md"
V9_BRANCH_LABELS_PATH = OUTPUTS_V9_DIR / "branch_labels.parquet"
V9_BRANCH_FEATURES_PATH = OUTPUTS_V9_DIR / "branch_features_v9.parquet"
V9_BRANCH_FEATURES_ENRICHED_PATH = OUTPUTS_V9_DIR / "branch_features_v9_enriched.parquet"
V9_PERSONA_CALIBRATION_HISTORY_PATH = OUTPUTS_V9_DIR / "persona_calibration_history.parquet"
V9_MEMORY_BANK_REPORT_PATH = OUTPUTS_V9_DIR / "memory_bank_report.json"
V9_CONTRADICTION_REPORT_PATH = OUTPUTS_V9_DIR / "contradiction_report.json"
V9_SELECTOR_CHECKPOINT_PATH = V9_CHECKPOINT_DIR / "selector_torch.pt"
V9_MEMORY_BANK_DIR = V9_CHECKPOINT_DIR / "memory_bank"
V9_MEMORY_BANK_ENCODER_PATH = V9_MEMORY_BANK_DIR / "encoder.pt"
V9_MEMORY_BANK_INDEX_PATH = V9_MEMORY_BANK_DIR / "index.npz"
V9_CONTRADICTION_MODEL_PATH = V9_CHECKPOINT_DIR / "contradiction_detector.pkl"
V9_SELECTOR_RESULTS_PATH = OUTPUTS_V9_DIR / "selector_experiment_results.json"
V9_SELECTOR_RESULTS_MD_PATH = OUTPUTS_V9_DIR / "selector_experiment_results.md"
V9_SUMMARY_JSON_PATH = OUTPUTS_EVAL_DIR / "v9_summary.json"
V9_SUMMARY_MD_PATH = OUTPUTS_EVAL_DIR / "v9_summary.md"
V10_BRANCH_AUDIT_PATH = OUTPUTS_V10_DIR / "branch_diversity_audit.json"
V10_BRANCH_AUDIT_MD_PATH = OUTPUTS_V10_DIR / "branch_diversity_audit.md"
V10_BRANCH_ARCHIVE_PATH = OUTPUTS_V10_DIR / "branch_archive_v10.parquet"
V10_BRANCH_ARCHIVE_REPORT_PATH = OUTPUTS_V10_DIR / "branch_archive_v10.report.json"
V10_BRANCH_FEATURES_PATH = OUTPUTS_V10_DIR / "branch_features_v10.parquet"
V10_BRANCH_LABELS_PATH = OUTPUTS_V10_DIR / "branch_labels_v10.parquet"
V10_SELECTOR_CHECKPOINT_PATH = V10_CHECKPOINT_DIR / "selector_torch_v10.pt"
V10_SELECTOR_RESULTS_PATH = OUTPUTS_V10_DIR / "selector_experiment_results_v10.json"
V10_SELECTOR_RESULTS_MD_PATH = OUTPUTS_V10_DIR / "selector_experiment_results_v10.md"
V10_SUMMARY_JSON_PATH = OUTPUTS_EVAL_DIR / "v10_summary.json"
V10_SUMMARY_MD_PATH = OUTPUTS_EVAL_DIR / "v10_summary.md"
V11_RESEARCH_BACKTEST_PATH = OUTPUTS_V11_DIR / "research_backtest.json"
V11_RESEARCH_BACKTEST_MD_PATH = OUTPUTS_V11_DIR / "research_backtest.md"
V11_SUMMARY_JSON_PATH = OUTPUTS_EVAL_DIR / "v11_summary.json"
V11_SUMMARY_MD_PATH = OUTPUTS_EVAL_DIR / "v11_summary.md"
V11_SETL_MODEL_PATH = V11_CHECKPOINT_DIR / "setl_regressor.pkl"
V11_PCOP_STAGE5_MODEL_PATH = V11_CHECKPOINT_DIR / "pcop_stage5.pkl"
V11_PCOP_STAGE10_MODEL_PATH = V11_CHECKPOINT_DIR / "pcop_stage10.pkl"
V11_SELECTOR_MODEL_PATH = V11_CHECKPOINT_DIR / "selector_ranker.pkl"
MACRO_REPORT_PATH = OUTPUTS_EVAL_DIR / "macro_report.json"
QUANT_REPORT_PATH = OUTPUTS_EVAL_DIR / "quant_report.json"
MARKET_DYNAMICS_REPORT_PATH = OUTPUTS_EVAL_DIR / "market_dynamics_report.json"
NEWS_REPORT_PATH = OUTPUTS_EVAL_DIR / "news_report.json"
CROWD_REPORT_PATH = OUTPUTS_EVAL_DIR / "crowd_report.json"
PROBABILITY_CONE_HTML_PATH = OUTPUTS_CHARTS_DIR / "probability_cone.html"
PERSONA_BREAKDOWN_HTML_PATH = OUTPUTS_CHARTS_DIR / "persona_breakdown.html"
FINAL_DASHBOARD_HTML_PATH = OUTPUTS_CHARTS_DIR / "nexus_dashboard.html"
MODEL_SERVICE_HOST = os.getenv("NEXUS_MODEL_HOST", "0.0.0.0")
MODEL_SERVICE_PORT = int(os.getenv("NEXUS_MODEL_PORT", "8000"))
LLM_PROVIDER_DEFAULT = os.getenv("NEXUS_LLM_PROVIDER", "lm_studio")
LM_STUDIO_BASE_URL = os.getenv("NEXUS_LM_STUDIO_BASE_URL", "http://127.0.0.1:1234")
LM_STUDIO_MODEL = os.getenv("NEXUS_LM_STUDIO_MODEL", "openai/gpt-oss-20b")
LM_STUDIO_TIMEOUT_SECONDS = int(os.getenv("NEXUS_LM_STUDIO_TIMEOUT_SECONDS", "45"))
LM_STUDIO_ENABLED = os.getenv("NEXUS_LM_STUDIO_ENABLED", "1") == "1"
OLLAMA_BASE_URL = os.getenv("NEXUS_OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("NEXUS_OLLAMA_MODEL", "minimax-m2.7:cloud")
OLLAMA_TIMEOUT_SECONDS = int(os.getenv("NEXUS_OLLAMA_TIMEOUT_SECONDS", "60"))
OLLAMA_ENABLED = os.getenv("NEXUS_OLLAMA_ENABLED", "1") == "1"
V10_DIVERSE_BRANCHING_ENABLED = os.getenv("NEXUS_V10_DIVERSE_BRANCHING", "0") == "1"

NORM_STATS_PATH = PROJECT_ROOT / "config" / "norm_stats_1m.json"
PERSONA_CONFIG_PATH = PROJECT_ROOT / "config" / "persona_config.json"
DATASET_MANIFEST_PATH = PROJECT_ROOT / "config" / "dataset_manifest.json"
DOWNLOAD_REPORT_PATH = OUTPUTS_LOGS_DIR / "download_report.json"


def get_data_path(*parts: str) -> Path:
    return DATA_DIR.joinpath(*parts)


def get_local_path(*parts: str) -> Path:
    return LOCAL_PROJECT_ROOT.joinpath(*parts)


def get_project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)
