from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "Beijing_Multi_Site_Air_Quality"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DATASET_PATH = PROCESSED_DATA_DIR / "air_quality_cleaned.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
EDA_RESULTS_DIR = RESULTS_DIR / "eda"
MODELS_DIR = PROJECT_ROOT / "models"
LSTM_MODEL_PATH = MODELS_DIR / "lstm_pm25.keras"
LSTM_RESULTS_DIR = RESULTS_DIR / "lstm"
RANDOM_SEED = 42

EXPECTED_COLUMNS = [
    "No",
    "year",
    "month",
    "day",
    "hour",
    "PM2.5",
    "PM10",
    "SO2",
    "NO2",
    "CO",
    "O3",
    "TEMP",
    "PRES",
    "DEWP",
    "RAIN",
    "wd",
    "WSPM",
    "station",
]

REQUIRED_NUMERIC_COLUMNS = [
    "PM2.5",
    "PM10",
    "SO2",
    "NO2",
    "CO",
    "O3",
    "TEMP",
    "PRES",
    "DEWP",
    "RAIN",
    "WSPM",
]

TEMPORAL_FEATURE_COLUMNS = REQUIRED_NUMERIC_COLUMNS
TARGET_COLUMN = "PM2.5"
SEQUENCE_LENGTH = 24
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
