from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import PROCESSED_DATASET_PATH, RAW_DATA_DIR
from src.data_loader import load_all_station_data
from src.preprocessing import clean_air_quality_data, save_cleaned_dataset
from src.visualization import run_eda_visualizations


def main() -> None:
    raw_data = load_all_station_data(RAW_DATA_DIR)
    cleaned = clean_air_quality_data(raw_data)
    save_cleaned_dataset(cleaned, PROCESSED_DATASET_PATH)

    print(f"Stations detected: {cleaned['station'].nunique()}")
    print(f"Rows before cleaning: {len(raw_data):,}")
    print(f"Rows after cleaning: {len(cleaned):,}")
    print(f"Processed dataset saved to: {PROCESSED_DATASET_PATH}")

    plot_paths = run_eda_visualizations(cleaned, ROOT / 'results' / 'eda', raw_data=raw_data)
    for name, path in plot_paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
