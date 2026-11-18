from __future__ import annotations

from pathlib import Path
from typing import List, Union

import pandas as pd

from src.config import RAW_DATA_DIR


def list_station_files(raw_dir: Union[str, Path] = RAW_DATA_DIR) -> List[Path]:
    """Return all station CSV files from the Beijing multi-site raw directory."""
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_path}")

    files = sorted(raw_path.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {raw_path}")
    return files


def load_station_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load a single monitoring-station CSV file."""
    csv_path = Path(file_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Station file not found: {csv_path}")

    dataframe = pd.read_csv(csv_path)
    return dataframe


def load_all_station_data(raw_dir: Union[str, Path] = RAW_DATA_DIR) -> pd.DataFrame:
    """Load all station CSV files and combine them into one dataframe."""
    files = list_station_files(raw_dir)
    station_dataframes = [load_station_data(file_path) for file_path in files]
    combined = pd.concat(station_dataframes, ignore_index=True)
    combined["station"] = combined["station"].fillna("Unknown").astype(str)
    return combined


def get_dataset_summary(dataframe: pd.DataFrame) -> dict:
    """Return a compact summary useful during inspection."""
    summary = {
        "rows": int(len(dataframe)),
        "columns": list(dataframe.columns),
        "stations": sorted(dataframe["station"].dropna().unique().tolist()),
        "date_min": dataframe.get("datetime").min() if "datetime" in dataframe.columns else None,
        "date_max": dataframe.get("datetime").max() if "datetime" in dataframe.columns else None,
    }
    return summary
