"""Prepare chronological input sequences for PM2.5 forecasting."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import (
    SEQUENCE_LENGTH,
    RAW_DATA_DIR,
    TARGET_COLUMN,
    TEMPORAL_FEATURE_COLUMNS,
    TRAIN_FRACTION,
    VALIDATION_FRACTION,
)
from src.data_loader import load_all_station_data
from src.preprocessing import build_datetime_column, validate_columns


def load_raw_temporal_data(raw_dir=RAW_DATA_DIR) -> pd.DataFrame:
    """Load station files and prepare timestamps without fitting imputations."""
    raw_data = load_all_station_data(raw_dir)
    validate_columns(raw_data)
    temporal_data = build_datetime_column(raw_data)
    temporal_data = temporal_data.sort_values(["station", "datetime"])
    temporal_data = temporal_data.drop_duplicates(
        subset=["station", "datetime"],
        keep="first",
    )
    if temporal_data.empty:
        raise ValueError("No valid timestamped station records were found.")
    return temporal_data.reset_index(drop=True)


def create_sequences(
    dataframe: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
    sequence_length: int = SEQUENCE_LENGTH,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Create station-specific sliding windows without crossing time gaps."""
    if sequence_length <= 0 or horizon <= 0:
        raise ValueError("sequence_length and horizon must be greater than zero.")

    required_columns = {"station", "datetime", target_column, *feature_columns}
    missing_columns = sorted(required_columns.difference(dataframe.columns))
    if missing_columns:
        raise ValueError(f"Missing columns required for sequence creation: {missing_columns}")

    input_sequences = []
    target_values = []

    for _, station_data in dataframe.groupby("station", sort=True):
        station_data = station_data.sort_values("datetime")
        if station_data["datetime"].duplicated().any():
            raise ValueError("Duplicate timestamps within a station must be removed before sequence creation.")
        features = station_data[feature_columns].to_numpy(dtype=np.float32)
        targets = station_data[target_column].to_numpy(dtype=np.float32)
        if not np.isfinite(features).all():
            raise ValueError("Feature values must be finite before sequence creation.")
        timestamps = pd.to_datetime(station_data["datetime"]).reset_index(drop=True)

        first_target_index = sequence_length + horizon - 1
        for target_index in range(first_target_index, len(station_data)):
            window_start = target_index - horizon - sequence_length + 1
            expected_gap = pd.Timedelta(hours=sequence_length + horizon - 1)
            if timestamps.iloc[target_index] - timestamps.iloc[window_start] != expected_gap:
                continue
            if not np.isfinite(targets[target_index]):
                continue

            window_end = target_index - horizon + 1
            input_sequences.append(features[window_start:window_end])
            target_values.append(targets[target_index])

    if not input_sequences:
        return (
            np.empty((0, sequence_length, len(feature_columns)), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    return np.asarray(input_sequences, dtype=np.float32), np.asarray(target_values, dtype=np.float32)


def prepare_temporal_data(
    dataframe: pd.DataFrame,
    feature_columns: List[str] = TEMPORAL_FEATURE_COLUMNS,
    target_column: str = TARGET_COLUMN,
    sequence_length: int = SEQUENCE_LENGTH,
    train_fraction: float = TRAIN_FRACTION,
    validation_fraction: float = VALIDATION_FRACTION,
    horizon: int = 1,
) -> Dict[str, object]:
    """Split by timestamp, scale using training data only, and build sequences.

    Each split is sequenced independently. This keeps validation and test
    windows from borrowing observations from earlier partitions.
    """
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction plus validation_fraction must be less than 1.")
    if target_column not in feature_columns:
        raise ValueError("The target column must also be included in feature_columns.")

    required_columns = {"station", "datetime", target_column, *feature_columns}
    missing_columns = sorted(required_columns.difference(dataframe.columns))
    if missing_columns:
        raise ValueError(f"Missing columns required for temporal preparation: {missing_columns}")
    if dataframe.empty:
        raise ValueError("The input dataframe is empty.")

    prepared = dataframe[["station", "datetime", *feature_columns]].copy()
    prepared["datetime"] = pd.to_datetime(prepared["datetime"], errors="coerce")
    if prepared["datetime"].isna().any():
        raise ValueError("datetime contains invalid or missing values.")
    if prepared["station"].isna().any():
        raise ValueError("station contains missing values.")
    if prepared.duplicated(["station", "datetime"]).any():
        raise ValueError("Duplicate station/datetime records must be removed before sequence creation.")

    for column in feature_columns:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    prepared = prepared.sort_values(["datetime", "station"]).reset_index(drop=True)
    timestamps = np.sort(prepared["datetime"].unique())
    train_end = int(len(timestamps) * train_fraction)
    validation_end = int(len(timestamps) * (train_fraction + validation_fraction))
    if train_end <= sequence_length + horizon - 1:
        raise ValueError("The training period is too short for the requested sequence length and horizon.")
    if validation_end - train_end <= sequence_length + horizon - 1:
        raise ValueError("The validation period is too short for the requested sequence length and horizon.")
    if len(timestamps) - validation_end <= sequence_length + horizon - 1:
        raise ValueError("The test period is too short for the requested sequence length and horizon.")

    train_times = timestamps[:train_end]
    validation_times = timestamps[train_end:validation_end]
    test_times = timestamps[validation_end:]
    train_data = prepared[prepared["datetime"].isin(train_times)].copy()
    validation_data = prepared[prepared["datetime"].isin(validation_times)].copy()
    test_data = prepared[prepared["datetime"].isin(test_times)].copy()

    feature_medians = train_data[feature_columns].median()
    if feature_medians.isna().any():
        empty_columns = feature_medians[feature_medians.isna()].index.tolist()
        raise ValueError(f"Training data has no values to impute these features: {empty_columns}")

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    train_features = train_data[feature_columns].fillna(feature_medians)
    train_targets = train_data.loc[train_data[target_column].notna(), [target_column]]
    if train_targets.empty:
        raise ValueError("The training period has no valid target observations.")
    feature_scaler.fit(train_features)
    target_scaler.fit(train_targets)

    def make_scaled_sequences(partition: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        scaled = partition.copy()
        partition_features = partition[feature_columns].fillna(feature_medians)
        scaled.loc[:, feature_columns] = feature_scaler.transform(partition_features)
        scaled["_scaled_target"] = target_scaler.transform(partition[[target_column]]).ravel()
        return create_sequences(
            scaled,
            feature_columns=feature_columns,
            target_column="_scaled_target",
            sequence_length=sequence_length,
            horizon=horizon,
        )

    X_train, y_train = make_scaled_sequences(train_data)
    X_validation, y_validation = make_scaled_sequences(validation_data)
    X_test, y_test = make_scaled_sequences(test_data)

    if not len(y_train) or not len(y_validation) or not len(y_test):
        raise ValueError("At least one chronological split has no valid target sequences.")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_validation": X_validation,
        "y_validation": y_validation,
        "X_test": X_test,
        "y_test": y_test,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "feature_medians": feature_medians,
        "split_timestamps": {
            "train_end": pd.Timestamp(train_times[-1]),
            "validation_start": pd.Timestamp(validation_times[0]),
            "validation_end": pd.Timestamp(validation_times[-1]),
            "test_start": pd.Timestamp(test_times[0]),
        },
    }
