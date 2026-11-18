from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

from src.config import EXPECTED_COLUMNS, PROCESSED_DATASET_PATH, RAW_DATA_DIR, REQUIRED_NUMERIC_COLUMNS
from src.data_loader import load_all_station_data


def validate_columns(dataframe: pd.DataFrame) -> None:
    """Validate that the raw data contains the expected air-quality fields."""
    missing = [column for column in EXPECTED_COLUMNS if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def build_datetime_column(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Create a single datetime column from year/month/day/hour."""
    cleaned = dataframe.copy()
    cleaned["year"] = pd.to_numeric(cleaned["year"], errors="coerce")
    cleaned["month"] = pd.to_numeric(cleaned["month"], errors="coerce")
    cleaned["day"] = pd.to_numeric(cleaned["day"], errors="coerce")
    cleaned["hour"] = pd.to_numeric(cleaned["hour"], errors="coerce")

    valid_mask = cleaned[["year", "month", "day", "hour"]].notna().all(axis=1)
    cleaned = cleaned.loc[valid_mask].copy()
    cleaned["datetime"] = pd.to_datetime(
        cleaned[["year", "month", "day", "hour"]].astype(int),
        errors="coerce",
    )
    cleaned = cleaned.loc[cleaned["datetime"].notna()].copy()
    return cleaned


def handle_missing_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Fill missing numeric values using station-wise medians to preserve station context."""
    cleaned = dataframe.copy()
    cleaned["wd"] = cleaned["wd"].fillna("Unknown")

    for station_name, station_df in cleaned.groupby("station"):
        for column in REQUIRED_NUMERIC_COLUMNS:
            if column not in cleaned.columns:
                continue
            station_median = station_df[column].median()
            if pd.notna(station_median):
                cleaned.loc[cleaned["station"] == station_name, column] = (
                    cleaned.loc[cleaned["station"] == station_name, column].fillna(station_median)
                )

    return cleaned


def clean_air_quality_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Run the data validation and cleaning steps for the Beijing air-quality dataset."""
    if dataframe.empty:
        raise ValueError("The input dataframe is empty.")

    validate_columns(dataframe)
    dataframe = dataframe.copy()
    dataframe["station"] = dataframe["station"].fillna("Unknown").astype(str)
    dataframe = build_datetime_column(dataframe)
    dataframe = dataframe.sort_values(["station", "datetime"]).reset_index(drop=True)
    dataframe = dataframe.drop_duplicates(subset=["station", "datetime"], keep="first").reset_index(drop=True)
    dataframe = handle_missing_values(dataframe)
    return dataframe


def save_cleaned_dataset(dataframe: pd.DataFrame, output_path: Union[str, Path] = PROCESSED_DATASET_PATH) -> Path:
    """Save the cleaned dataset to the project processed-data directory."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output, index=False)
    return output


def prepare_clean_dataset(raw_dir: Union[str, Path] = RAW_DATA_DIR) -> pd.DataFrame:
    """Convenience wrapper to load the raw station files and build the cleaned dataset."""
    raw_data = load_all_station_data(raw_dir)
    cleaned = clean_air_quality_data(raw_data)
    save_cleaned_dataset(cleaned)
    return cleaned
