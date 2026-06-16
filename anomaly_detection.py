"""
Deep Learning-Based Anomaly Detection on S&P 500 Time Series
-------------------------------------------------------------
Uses an autoencoder (LSTM-style dense encoder-decoder) trained on normal
S&P 500 closing price data to flag anomalous market behaviour.

Run:  python anomaly_detection.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
)
import tensorflow as tf
import yfinance as yf


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class AnomalyDetector(tf.keras.Model):
    """
    Autoencoder-based anomaly detector.

    Encodes input to a compressed 32-dimensional representation, then
    reconstructs it. High reconstruction error → likely anomaly.
    """

    def __init__(self):
        super(AnomalyDetector, self).__init__()
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64,  activation="relu"),
            tf.keras.layers.Dense(32,  activation="relu"),
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(64,  activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(1,   activation="sigmoid"),
        ])

    def call(self, x):
        return self.decoder(self.encoder(x))


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def load_sp500(period: str = "max") -> np.ndarray:
    """Download S&P 500 closing prices and return as a numpy array."""
    ticker = yf.Ticker("^GSPC")
    data = ticker.history(period=period)
    series = data["Close"].rolling(4).sum().dropna()
    return series.values


def normalise(data: np.ndarray) -> tuple[np.ndarray, MinMaxScaler]:
    """Min-max scale data to [0, 1]. Returns (scaled_array, fitted_scaler)."""
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data.reshape(-1, 1))
    return scaled, scaler


def split_data(
    data: np.ndarray,
    train_frac: float = 0.7,
    val_frac: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train / validation / test sets.

    Args:
        data:       1-D or 2-D numpy array.
        train_frac: Fraction for training.
        val_frac:   Fraction for validation. Test gets the remainder.

    Returns:
        (X_train, X_val, X_test)
    """
    n = len(data)
    train_end = int(train_frac * n)
    val_end   = train_end + int(val_frac * n)
    return data[:train_end], data[train_end:val_end], data[val_end:]


def predict_anomalies(
    model: tf.keras.Model,
    data: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Predict anomalies based on reconstruction error.

    Args:
        model:     Trained autoencoder.
        data:      Input data array.
        threshold: MAE threshold above which a point is flagged as anomalous.

    Returns:
        Boolean numpy array — True where anomaly is detected.
    """
    reconstructions = model.predict(data, verbose=0)
    loss = tf.keras.losses.mean_absolute_error(reconstructions, data).numpy()
    return loss > threshold


def compute_metrics(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    anomaly_scores: np.ndarray,
) -> dict:
    """
    Compute binary classification metrics.

    Args:
        true_labels:      Ground-truth binary labels (0 = normal, 1 = anomaly).
        predicted_labels: Binary predictions from the model.
        anomaly_scores:   Continuous anomaly scores (reconstruction losses).

    Returns:
        Dictionary with accuracy, precision, recall, f1, auc keys.
    """
    fpr, tpr, _ = roc_curve(true_labels, anomaly_scores)
    return {
        "accuracy":  accuracy_score(true_labels, predicted_labels),
        "precision": precision_score(true_labels, predicted_labels, zero_division=0),
        "recall":    recall_score(true_labels, predicted_labels, zero_division=0),
        "f1":        f1_score(true_labels, predicted_labels, zero_division=0),
        "auc":       auc(fpr, tpr),
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    # --- Data loading and preprocessing ---
    print("Downloading S&P 500 data...")
    raw = load_sp500()
    scaled, scaler = normalise(raw)
    X_train, X_val, X_test = split_data(scaled)

    print(f"Split sizes — Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # --- Model ---
    autoencoder = AnomalyDetector()
    autoencoder.compile(optimizer="adam", loss="mae")

    print("Training autoencoder...")
    history = autoencoder.fit(
        X_train, X_train,
        epochs=100,
        validation_data=(X_val, X_val),
        batch_size=64,
        verbose=1,
    )

    # --- Evaluation ---
    test_loss = autoencoder.evaluate(X_test, X_test, verbose=0)
    print(f"\nTest reconstruction loss (MAE): {test_loss:.4f}")

    # Reconstruction losses on test set (used as anomaly scores)
    reconstructions = autoencoder.predict(X_test, verbose=0)
    anomaly_scores = tf.keras.losses.mean_absolute_error(
        reconstructions, X_test
    ).numpy()

    # Threshold-based binary predictions
    threshold = 0.15
    predicted_labels = (anomaly_scores > threshold).astype(int)

    # Since S&P 500 has no labelled anomalies, we treat all test points as
    # normal (0) and report the false-positive rate at our chosen threshold.
    true_labels = np.zeros(len(X_test), dtype=int)

    metrics = compute_metrics(true_labels, predicted_labels, anomaly_scores)
    print("\nPerformance Metrics:")
    for name, value in metrics.items():
        print(f"  {name.capitalize()}: {value:.4f}")

    # --- Plots ---
    plt.figure(figsize=(10, 4))
    plt.plot(history.history["loss"],     label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MAE Loss")
    plt.title("Autoencoder Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_loss.png", dpi=150)
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(anomaly_scores, label="Reconstruction Error")
    plt.axhline(threshold, color="red", linestyle="--", label=f"Threshold ({threshold})")
    plt.xlabel("Test Sample Index")
    plt.ylabel("MAE")
    plt.title("Anomaly Detection on S&P 500 Test Set")
    plt.legend()
    plt.tight_layout()
    plt.savefig("anomaly_scores.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
