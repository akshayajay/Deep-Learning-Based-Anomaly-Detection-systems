# Deep Learning-Based Anomaly Detection

[![Tests](https://github.com/akshayajay/Deep-Learning-Based-Anomaly-Detection-systems/actions/workflows/tests.yml/badge.svg)](https://github.com/akshayajay/Deep-Learning-Based-Anomaly-Detection-systems/actions/workflows/tests.yml)

Detecting anomalous market behaviour in S&P 500 time series using a deep autoencoder — trained only on normal data, flagging points where reconstruction error exceeds a threshold.

---

## How it works

An autoencoder learns to compress and reconstruct normal S&P 500 closing price sequences. When the model encounters an unusual pattern it wasn't trained on, reconstruction error spikes — that spike is the anomaly signal.

```
Input → Encoder (128→64→32) → Decoder (32→64→128→1) → Reconstruction
                                          ↓
                               MAE(input, reconstruction) > threshold → Anomaly
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/akshayajay/Deep-Learning-Based-Anomaly-Detection-systems.git
cd Deep-Learning-Based-Anomaly-Detection-systems
pip install -r requirements.txt
```

### 2. Run

```bash
python anomaly_detection.py
```

Data is fetched automatically from Yahoo Finance. Two plots are saved: `training_loss.png` and `anomaly_scores.png`.

### 3. Run tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
├── anomaly_detection.py       # Main script
├── tests/
│   └── test_anomaly_detection.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Tech Stack

`Python` · `TensorFlow/Keras` · `scikit-learn` · `yfinance` · `NumPy` · `matplotlib`

---

## Author

**Akshaya J** · [github.com/akshayajay](https://github.com/akshayajay)
