"""Train the PM2.5 LSTM using chronological training and validation data."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from src.config import (
    LSTM_MODEL_PATH,
    LSTM_RESULTS_DIR,
    RANDOM_SEED,
    SEQUENCE_LENGTH,
    TEMPORAL_FEATURE_COLUMNS,
)
from src.lstm_model import build_lstm_model
from src.temporal_data import load_raw_temporal_data, prepare_temporal_data


def train_lstm(
    epochs: int = 30,
    batch_size: int = 256,
    patience: int = 5,
    sequence_length: int = SEQUENCE_LENGTH,
    model_path: Path = LSTM_MODEL_PATH,
    results_dir: Path = LSTM_RESULTS_DIR,
) -> Dict[str, object]:
    """Train and save the best validation-loss LSTM checkpoint."""
    if epochs <= 0 or batch_size <= 0 or patience <= 0:
        raise ValueError("epochs, batch_size, and patience must be greater than zero.")

    tf.keras.utils.set_random_seed(RANDOM_SEED)
    temporal_data = prepare_temporal_data(
        load_raw_temporal_data(),
        sequence_length=sequence_length,
    )
    model = build_lstm_model(
        sequence_length=sequence_length,
        n_features=len(TEMPORAL_FEATURE_COLUMNS),
    )

    return fit_lstm_model(
        model=model,
        temporal_data=temporal_data,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        model_path=model_path,
        results_dir=results_dir,
    )


def fit_lstm_model(
    model: tf.keras.Model,
    temporal_data: Dict[str, object],
    epochs: int,
    batch_size: int,
    patience: int,
    model_path: Path,
    results_dir: Path,
) -> Dict[str, object]:
    """Fit a compiled LSTM from prepared arrays and save its best checkpoint."""
    if epochs <= 0 or batch_size <= 0 or patience <= 0:
        raise ValueError("epochs, batch_size, and patience must be greater than zero.")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="mean_squared_error",
        metrics=["mean_absolute_error"],
    )

    model_path = Path(model_path)
    results_dir = Path(results_dir)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        temporal_data["X_train"],
        temporal_data["y_train"],
        validation_data=(
            temporal_data["X_validation"],
            temporal_data["y_validation"],
        ),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        callbacks=callbacks,
        verbose=2,
    )

    model.save(model_path)
    history_frame = pd.DataFrame(history.history)
    history_frame.to_csv(results_dir / "training_history.csv", index=False)
    _plot_training_history(history_frame, results_dir / "training_loss.png")

    return {
        "model_path": model_path,
        "history": history_frame,
        "training_sequences": len(temporal_data["y_train"]),
        "validation_sequences": len(temporal_data["y_validation"]),
    }


def _plot_training_history(history: pd.DataFrame, output_path: Path) -> None:
    """Save training and validation loss curves."""
    plt.figure(figsize=(9, 5))
    plt.plot(history["loss"], label="Training loss")
    plt.plot(history["val_loss"], label="Validation loss")
    plt.title("LSTM Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Mean squared error (scaled PM2.5)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the PM2.5 LSTM model.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH)
    parser.add_argument("--model-path", type=Path, default=LSTM_MODEL_PATH)
    parser.add_argument("--results-dir", type=Path, default=LSTM_RESULTS_DIR)
    arguments = parser.parse_args()

    result = train_lstm(
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        patience=arguments.patience,
        sequence_length=arguments.sequence_length,
        model_path=arguments.model_path,
        results_dir=arguments.results_dir,
    )
    print(f"Best LSTM model saved to: {result['model_path']}")
    print(f"Training sequences: {result['training_sequences']:,}")
    print(f"Validation sequences: {result['validation_sequences']:,}")
    print(f"Training history and loss plot saved in: {LSTM_RESULTS_DIR}")


if __name__ == "__main__":
    main()
