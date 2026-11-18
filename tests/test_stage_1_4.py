from pathlib import Path

from src.config import RAW_DATA_DIR
from src.data_loader import list_station_files, load_all_station_data
from src.preprocessing import clean_air_quality_data


def test_station_csv_files_are_detected():
    station_files = list_station_files(RAW_DATA_DIR)
    assert len(station_files) == 12
    assert all(file_path.suffix == ".csv" for file_path in station_files)


def test_combined_air_quality_data_has_expected_columns():
    combined = load_all_station_data(RAW_DATA_DIR)
    assert not combined.empty
    assert combined["station"].nunique() == 12
    assert {"year", "month", "day", "hour", "PM2.5", "station"}.issubset(set(combined.columns))


def test_cleaning_creates_valid_datetime_and_processes_missing_values():
    raw = load_all_station_data(RAW_DATA_DIR)
    cleaned = clean_air_quality_data(raw)

    assert not cleaned.empty
    assert "datetime" in cleaned.columns
    assert cleaned["datetime"].notna().all()
    assert cleaned["station"].nunique() == 12
