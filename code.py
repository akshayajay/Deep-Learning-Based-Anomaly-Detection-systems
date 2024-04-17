import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import yfinance as yf
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

# Define the AnomalyDetector class
class AnomalyDetector(tf.keras.Model):
    def __init__(self):
        super(AnomalyDetector, self).__init__()
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu')
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
    
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Get historical market data
SP500 = yf.Ticker("^GSPC")
SP500_data = SP500.history(period="max")
SP500_data = SP500_data.Close.rolling(4).sum().dropna()
SP500_data.index = SP500_data.index.strftime('%m/%d/%Y')

# Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(np.array(SP500_data).reshape(-1, 1))

# Split the data into training, validation, and test sets
train_size = int(0.7 * len(scaled_data))
test_size = int(0.1 * len(scaled_data))
val_size = int(0.2 * len(scaled_data))
X_train = scaled_data[:train_size]
X_test = scaled_data[train_size:train_size+test_size]
X_val = scaled_data[train_size+test_size:train_size+test_size+val_size]

# Create the anomaly detector model
autoencoder = AnomalyDetector()

# Compile the model
autoencoder.compile(optimizer='adam', loss='mae')

# Train the model
history = autoencoder.fit(X_train, X_train, epochs=100, validation_data=(X_val, X_val), batch_size=64)

# Evaluate the model on the test set
test_loss = autoencoder.evaluate(X_test, X_test)

# Generate predictions on the test set
reconstructions = autoencoder.predict(X_test)

# Calculate the loss between the test set and the reconstructions
test_loss = tf.keras.losses.mean_absolute_error(reconstructions, X_test)

# Define the function to predict anomalies
def predict_anomalies(model, data, threshold):
    reconstructions = model.predict(data)
    loss = tf.keras.losses.mean_absolute_error(reconstructions, data)
    anomalies = loss > threshold
    return anomalies

# Set the threshold for anomaly detection
threshold = 0.15  

# Predict anomalies on the test set
test_anomalies = predict_anomalies(autoencoder, X_test, threshold)

# Generate true labels for the test set (assuming all points are normal)
true_labels = np.zeros_like(test_anomalies, dtype=bool)

# Assuming true_labels and predicted_labels are available
true_labels = np.array([0, 1, 0, 1, 0])  # Example ground truth labels
predicted_labels = np.array([0, 1, 1, 1, 0])  # Example predicted labels

# Calculate accuracy
accuracy = accuracy_score(true_labels, predicted_labels)

# Calculate precision
precision = precision_score(true_labels, predicted_labels)

# Calculate recall
recall = recall_score(true_labels, predicted_labels)

# Calculate F1-score
f1 = f1_score(true_labels, predicted_labels)

# Assuming anomaly_scores are available for ROC curve calculation
anomaly_scores = np.array([0.1, 0.8, 0.3, 0.9, 0.2])  # Example anomaly scores

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(true_labels, anomaly_scores)
auc_score = auc(fpr, tpr)


# Print performance metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("AUC:", auc_score)

# Plot the training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
