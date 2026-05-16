import os
from pathlib import Path
from src.utils import BASE_DIR

# Project Roots
PROJECT_ROOT = BASE_DIR
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = MODEL_DIR / "artifacts"
TRAINED_MODELS_DIR = MODEL_DIR / "trained_models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Raw Data Files (IEEE-CIS Standard)
TRAIN_TRANSACTION = DATA_DIR / "train_transaction.csv"
TRAIN_IDENTITY = DATA_DIR / "train_identity.csv"

# Processed Data Files
PROCESSED_TRAIN_ORIGINAL = PROCESSED_DATA_DIR / "processed_train_original.parquet"
PROCESSED_TEST = PROCESSED_DATA_DIR / "processed_test.parquet"

# Preprocessing Constants
MISSING_VALUE_THRESHOLD = 0.85  # Drop columns with >85% missing — tree models handle sparse features well
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Feature Types
TARGET_COL = "isFraud"
ID_COL = "TransactionID"
TIME_COL = "TransactionDT"

# Categorical columns for Label Encoding
CAT_COLS = [
    'ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain',
    'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
    'id_12', 'id_15', 'id_16', 'id_28', 'id_29', 'id_30', 'id_31', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo'
]

# Artifact Paths
ENCODER_PATH = ARTIFACTS_DIR / "label_encoders.pkl"
IMPUTER_VALUES_PATH = ARTIFACTS_DIR / "imputer_values.pkl"
FEATURE_COLS_PATH = ARTIFACTS_DIR / "feature_columns.pkl"
CALIBRATION_MODEL_PATH = ARTIFACTS_DIR / "calibration_model.pkl"
FEATURE_STATS_PATH = ARTIFACTS_DIR / "training_feature_stats.json"

# Feature Names (To ensure consistency)
BEST_MODEL_PATH = TRAINED_MODELS_DIR / "best_model.pkl"
THRESHOLD_PATH = ARTIFACTS_DIR / "threshold.json"

# Output Paths
PREDICTIONS_PATH = OUTPUT_DIR / "predictions.parquet"
METRICS_PATH = OUTPUT_DIR / "evaluation_metrics.json"
FEATURE_IMPORTANCE_PATH = OUTPUT_DIR / "feature_importance.csv"
