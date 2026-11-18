"""Initial LSTM architecture for hourly PM2.5 forecasting."""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM


def build_lstm_model(sequence_length: int, n_features: int) -> Sequential:
    """Build the Stage 5 LSTM model; training and compilation happen later."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be greater than zero.")
    if n_features <= 0:
        raise ValueError("n_features must be greater than zero.")

    return Sequential(
        [
            Input(shape=(sequence_length, n_features)),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dense(32, activation="relu"),
            Dense(1),
        ],
        name="pm25_lstm",
    )
