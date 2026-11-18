import numpy as np
from sklearn.preprocessing import StandardScaler

from src.evaluate_lstm import calculate_regression_metrics, predict_pm25
from src.lstm_model import build_lstm_model
from src.train_lstm import fit_lstm_model


class FixedPredictionModel:
    def predict(self, sequences, batch_size, verbose):
        return np.array([[0.0], [1.0]], dtype=np.float32)


def test_prediction_is_converted_back_to_original_pm25_units():
    target_scaler = StandardScaler().fit(np.array([[10.0], [20.0]]))

    predictions = predict_pm25(
        FixedPredictionModel(),
        np.zeros((2, 3, 1), dtype=np.float32),
        target_scaler,
    )

    np.testing.assert_allclose(predictions, [15.0, 20.0])


def test_regression_metrics_are_calculated_correctly():
    metrics = calculate_regression_metrics(
        actual=np.array([1.0, 3.0]),
        predicted=np.array([1.0, 2.0]),
    )

    assert metrics["mae"] == 0.5
    assert metrics["rmse"] == np.sqrt(0.5)
    assert metrics["r2"] == 0.5


def test_training_smoke_saves_checkpoint_and_history(tmp_path):
    random = np.random.default_rng(42)
    temporal_data = {
        "X_train": random.normal(size=(8, 3, 2)).astype(np.float32),
        "y_train": random.normal(size=8).astype(np.float32),
        "X_validation": random.normal(size=(2, 3, 2)).astype(np.float32),
        "y_validation": random.normal(size=2).astype(np.float32),
    }
    model_path = tmp_path / "smoke_lstm.keras"
    results_dir = tmp_path / "results"

    result = fit_lstm_model(
        model=build_lstm_model(sequence_length=3, n_features=2),
        temporal_data=temporal_data,
        epochs=1,
        batch_size=4,
        patience=1,
        model_path=model_path,
        results_dir=results_dir,
    )

    assert model_path.exists()
    assert result["history"].shape[0] == 1
    assert (results_dir / "training_history.csv").exists()
    assert (results_dir / "training_loss.png").exists()
