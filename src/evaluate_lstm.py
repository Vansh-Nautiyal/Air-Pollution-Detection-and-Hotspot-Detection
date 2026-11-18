"""Predict and evaluate the saved PM2.5 LSTM on the held-out test period."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

from src.config import (
    LSTM_MODEL_PATH,
    LSTM_RESULTS_DIR,
    TARGET_COLUMN,
    TEMPORAL_FEATURE_COLUMNS,
)
from src.temporal_data import load_raw_temporal_data, prepare_temporal_data


def predict_pm25(
    model,
    sequences: np.ndarray,
    target_scaler: StandardScaler,
    batch_size: int = 256,
) -> np.ndarray:
    """Predict PM2.5 and convert model outputs back to their original units."""
    if len(sequences) == 0:
        raise ValueError("Cannot predict from an empty sequence array.")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    scaled_predictions = model.predict(sequences, batch_size=batch_size, verbose=0)
    return target_scaler.inverse_transform(
        np.asarray(scaled_predictions).reshape(-1, 1)
    ).ravel()


def calculate_regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """Calculate MAE, RMSE, and R-squared for PM2.5 predictions."""
    actual = np.asarray(actual).ravel()
    predicted = np.asarray(predicted).ravel()
    if actual.size == 0 or actual.shape != predicted.shape:
        raise ValueError("actual and predicted must be non-empty arrays with matching shapes.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("actual and predicted values must be finite.")

    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def evaluate_saved_model(
    model_path: Path = LSTM_MODEL_PATH,
    results_dir: Path = LSTM_RESULTS_DIR,
    batch_size: int = 256,
) -> Dict[str, object]:
    """Evaluate the saved checkpoint once on the chronological test set."""
    model_path = Path(model_path)
    results_dir = Path(results_dir)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Run python -m src.train_lstm first."
        )
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    model = load_model(model_path)
    input_shape = model.input_shape
    if len(input_shape) != 3 or input_shape[-1] != len(TEMPORAL_FEATURE_COLUMNS):
        raise ValueError(
            "The saved model input shape does not match the configured temporal features."
        )
    sequence_length = int(input_shape[1])
    temporal_data = prepare_temporal_data(
        load_raw_temporal_data(),
        sequence_length=sequence_length,
    )

    start_time = time.perf_counter()
    predicted = predict_pm25(
        model,
        temporal_data["X_test"],
        temporal_data["target_scaler"],
        batch_size=batch_size,
    )
    prediction_seconds = time.perf_counter() - start_time

    target_scaler = temporal_data["target_scaler"]
    actual = target_scaler.inverse_transform(
        temporal_data["y_test"].reshape(-1, 1)
    ).ravel()

    target_feature_index = TEMPORAL_FEATURE_COLUMNS.index(TARGET_COLUMN)
    last_observations = temporal_data["feature_scaler"].inverse_transform(
        temporal_data["X_test"][:, -1, :]
    )[:, target_feature_index]
    persistence_metrics = calculate_regression_metrics(actual, last_observations)

    results = {
        "test_samples": int(len(actual)),
        "sequence_length_hours": sequence_length,
        "forecast_horizon_hours": 1,
        "lstm": calculate_regression_metrics(actual, predicted),
        "persistence_baseline": persistence_metrics,
        "inference": {
            "total_seconds": float(prediction_seconds),
            "milliseconds_per_sample": float(prediction_seconds * 1000 / len(actual)),
        },
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "metrics.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "sample_index": np.arange(len(actual)),
            "actual_pm25": actual,
            "predicted_pm25": predicted,
            "persistence_baseline_pm25": last_observations,
        }
    ).to_csv(results_dir / "test_predictions.csv", index=False)
    _plot_actual_vs_predicted(
        actual,
        predicted,
        results_dir / "actual_vs_predicted.png",
    )
    return results


def _plot_actual_vs_predicted(
    actual: np.ndarray,
    predicted: np.ndarray,
    output_path: Path,
) -> None:
    """Save an actual-versus-predicted scatter plot with a perfect-fit line."""
    lower_bound = float(min(actual.min(), predicted.min()))
    upper_bound = float(max(actual.max(), predicted.max()))

    plt.figure(figsize=(7, 7))
    plt.scatter(actual, predicted, alpha=0.2, s=8)
    plt.plot(
        [lower_bound, upper_bound],
        [lower_bound, upper_bound],
        color="red",
        linestyle="--",
        label="Perfect prediction",
    )
    plt.title("Test Set: Actual vs Predicted PM2.5")
    plt.xlabel("Actual PM2.5 (μg/m³)")
    plt.ylabel("Predicted PM2.5 (μg/m³)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the trained PM2.5 LSTM.")
    parser.add_argument("--model-path", type=Path, default=LSTM_MODEL_PATH)
    parser.add_argument("--results-dir", type=Path, default=LSTM_RESULTS_DIR)
    parser.add_argument("--batch-size", type=int, default=256)
    arguments = parser.parse_args()

    results = evaluate_saved_model(
        model_path=arguments.model_path,
        results_dir=arguments.results_dir,
        batch_size=arguments.batch_size,
    )
    print(json.dumps(results, indent=2))
    print(f"Evaluation artifacts saved in: {LSTM_RESULTS_DIR}")


if __name__ == "__main__":
    main()
