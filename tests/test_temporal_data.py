import numpy as np
import pandas as pd

from src.lstm_model import build_lstm_model
from src.temporal_data import create_sequences, prepare_temporal_data


def _make_station_data(number_of_hours=30):
    timestamps = pd.date_range("2020-01-01", periods=number_of_hours, freq="h")
    records = []
    for station, offset in (("A", 0), ("B", 10)):
        for hour, timestamp in enumerate(timestamps):
            future_shift = 1000 if hour >= 18 else 0
            records.append(
                {
                    "station": station,
                    "datetime": timestamp,
                    "PM2.5": float(hour + offset + future_shift),
                    "TEMP": float(hour * 2 + offset),
                }
            )
    return pd.DataFrame(records)


def test_temporal_data_is_split_chronologically_and_scaled_from_training_only():
    data = prepare_temporal_data(
        _make_station_data(),
        feature_columns=["PM2.5", "TEMP"],
        sequence_length=3,
        train_fraction=0.6,
        validation_fraction=0.2,
    )

    assert data["X_train"].shape == (30, 3, 2)
    assert data["y_train"].shape == (30,)
    assert data["X_validation"].shape == (6, 3, 2)
    assert data["X_test"].shape == (6, 3, 2)
    assert data["feature_scaler"].mean_[0] == 13.5
    assert data["split_timestamps"]["train_end"] < data["split_timestamps"]["validation_start"]
    assert data["split_timestamps"]["validation_end"] < data["split_timestamps"]["test_start"]
    assert all(
        array.dtype == np.float32
        for array in (
            data["X_train"],
            data["y_train"],
            data["X_validation"],
            data["y_validation"],
            data["X_test"],
            data["y_test"],
        )
    )


def test_sequence_creation_does_not_bridge_missing_hourly_records():
    dataframe = _make_station_data(number_of_hours=10)
    dataframe = dataframe[
        ~((dataframe["station"] == "A") & (dataframe["datetime"] == pd.Timestamp("2020-01-01 05:00")))
    ]

    X, y = create_sequences(
        dataframe[dataframe["station"] == "A"],
        feature_columns=["PM2.5"],
        target_column="PM2.5",
        sequence_length=3,
    )

    assert X.shape == (3, 3, 1)
    assert y.tolist() == [3.0, 4.0, 9.0]


def test_missing_features_use_training_medians_and_missing_targets_are_skipped():
    dataframe = _make_station_data()
    dataframe.loc[
        (dataframe["station"] == "A")
        & (dataframe["datetime"] == pd.Timestamp("2020-01-01 18:00")),
        "PM2.5",
    ] = np.nan
    dataframe.loc[
        (dataframe["station"] == "A")
        & (dataframe["datetime"] == pd.Timestamp("2020-01-01 21:00")),
        "PM2.5",
    ] = np.nan

    data = prepare_temporal_data(
        dataframe,
        feature_columns=["PM2.5", "TEMP"],
        sequence_length=3,
        train_fraction=0.6,
        validation_fraction=0.2,
    )

    assert data["feature_medians"]["PM2.5"] == 13.5
    assert data["feature_scaler"].mean_[0] == 13.5
    assert data["X_validation"].shape == (5, 3, 2)
    assert np.isfinite(data["X_validation"]).all()
    assert np.isfinite(data["y_validation"]).all()


def test_lstm_architecture_returns_one_prediction_per_sequence():
    model = build_lstm_model(sequence_length=24, n_features=11)
    predictions = model(np.zeros((2, 24, 11), dtype=np.float32), training=False)

    assert tuple(predictions.shape) == (2, 1)
