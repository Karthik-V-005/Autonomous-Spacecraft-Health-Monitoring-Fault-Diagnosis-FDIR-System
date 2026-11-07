
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score
)


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

np.random.seed(42)
tf.random.set_seed(42)

TRAIN_PATH = "data/train/P-1.npy"
TEST_PATH = "data/test/P-1.npy"

WINDOW_SIZE = 10
BATCH_SIZE = 64
EPOCHS = 10

THRESHOLD_PERCENTILE = 99

# All three labelled P-1 anomaly intervals (inclusive)
ANOMALY_SEQUENCES = [
    (2149, 2349),
    (3539, 3779),
    (4536, 4844)
]


# --------------------------------------------------
# 2. Load telemetry data
# --------------------------------------------------

train_data = np.load(TRAIN_PATH)
test_data = np.load(TEST_PATH)

train_telemetry = train_data[:, 0].astype(np.float32)
test_telemetry = test_data[:, 0].astype(np.float32)

print("Training telemetry:", train_telemetry.shape)
print("Test telemetry:", test_telemetry.shape)


# --------------------------------------------------
# 3. Create sliding windows
# --------------------------------------------------

def create_windows(series, window_size):
    X, y = [], []

    for i in range(len(series) - window_size):
        X.append(series[i:i + window_size])
        y.append(series[i + window_size])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # LSTM input: (samples, timesteps, features)
    X = X.reshape(-1, window_size, 1)

    return X, y


# Chronological 80/20 training-validation split
split_idx = int(len(train_telemetry) * 0.8)

fit_series = train_telemetry[:split_idx]

# Retain preceding context for the first validation window
val_series = train_telemetry[split_idx - WINDOW_SIZE:]

X_train, y_train = create_windows(fit_series, WINDOW_SIZE)
X_val, y_val = create_windows(val_series, WINDOW_SIZE)
X_test, y_test = create_windows(test_telemetry, WINDOW_SIZE)

print("\nDataset shapes:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_val:", X_val.shape)
print("y_val:", y_val.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# --------------------------------------------------
# 4. Build and train the LSTM
# --------------------------------------------------

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(WINDOW_SIZE, 1)),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mse"
)

model.summary()

model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=False,
    validation_data=(X_val, y_val),
    verbose=1
)


# --------------------------------------------------
# 5. Generate predictions and calculate errors
# --------------------------------------------------

predictions = model.predict(X_test, verbose=0).reshape(-1)
val_predictions = model.predict(X_val, verbose=0).reshape(-1)

errors = np.abs(y_test - predictions)
val_errors = np.abs(y_val - val_predictions)

print("\n--- Error Distribution ---")

print("Validation percentiles:")
print(np.percentile(val_errors, [50, 90, 95, 99, 99.5, 100]))

print("\nTest percentiles:")
print(np.percentile(errors, [50, 90, 95, 99, 99.5, 100]))

print("\nValidation mean error:", np.mean(val_errors))
print("Test mean error:", np.mean(errors))

print("\n--- LSTM Forecasting Results ---")
print("Mean absolute error:", np.mean(errors))
print("Maximum absolute error:", np.max(errors))


# --------------------------------------------------
# 6. Calibrate anomaly threshold using validation data
# --------------------------------------------------

threshold = np.percentile(val_errors, THRESHOLD_PERCENTILE)

print(f"\n{THRESHOLD_PERCENTILE}th-percentile threshold:", threshold)


# --------------------------------------------------
# 7. Construct ground-truth labels for ALL anomalies
# --------------------------------------------------

# Each error corresponds to its target timestep
target_timesteps = np.arange(
    WINDOW_SIZE,
    len(test_telemetry)
)

ground_truth = np.zeros(len(target_timesteps), dtype=int)

for start, end in ANOMALY_SEQUENCES:
    anomaly_mask = (
        (target_timesteps >= start) &
        (target_timesteps <= end)
    )

    ground_truth[anomaly_mask] = 1


# --------------------------------------------------
# 8. Detect anomalies and evaluate performance
# --------------------------------------------------

predicted_labels = (errors > threshold).astype(int)

print("\n--- Anomaly Detection Results ---")
print("Threshold:", round(threshold, 6))

print("\nConfusion Matrix:")
print(confusion_matrix(ground_truth, predicted_labels))

print("\nClassification Report:")
print(classification_report(
    ground_truth,
    predicted_labels,
    target_names=["Normal", "Anomaly"],
    zero_division=0
))


# --------------------------------------------------
# 8A. Threshold-independent evaluation
# --------------------------------------------------

roc_auc = roc_auc_score(ground_truth, errors)

average_precision = average_precision_score(
    ground_truth,
    errors
)

print("\n--- Univariate LSTM Ranking Evaluation ---")
print("ROC-AUC:", round(roc_auc, 4))
print("Average Precision:", round(average_precision, 4))


# --------------------------------------------------
# 9. Compare normal and anomalous prediction errors
# --------------------------------------------------

normal_errors = errors[ground_truth == 0]
anomaly_errors = errors[ground_truth == 1]

print("\n--- Overall Error Comparison ---")

print("Mean error outside anomalies:", np.mean(normal_errors))
print("Mean error inside anomalies:", np.mean(anomaly_errors))

print("Maximum error outside anomalies:", np.max(normal_errors))
print("Maximum error inside anomalies:", np.max(anomaly_errors))


# --------------------------------------------------
# 10. Plot validation error distribution
# --------------------------------------------------

plt.figure(figsize=(10, 4))

plt.hist(val_errors, bins=60)

plt.axvline(
    threshold,
    color="red",
    linestyle="--",
    label=f"Threshold = {threshold:.3f}"
)

plt.xlabel("Absolute prediction error")
plt.ylabel("Frequency")
plt.title("P-1 Validation Error Distribution")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 11. Visualize prediction errors around first anomaly
# --------------------------------------------------

start = 2100
end = 2400

local_mask = (
    (target_timesteps >= start) &
    (target_timesteps <= end)
)

local_times = target_timesteps[local_mask]
local_errors = errors[local_mask]

plt.figure(figsize=(14, 5))

plt.plot(
    local_times,
    local_errors,
    label="LSTM prediction error",
    linewidth=1
)

plt.axvspan(
    2149,
    2349,
    color="orange",
    alpha=0.25,
    label="Ground-truth anomaly"
)

plt.axhline(
    threshold,
    color="red",
    linestyle="--",
    label=f"Threshold = {threshold:.3f}"
)

plt.xlabel("Timestep")
plt.ylabel("Absolute prediction error")
plt.title("P-1: Zoomed Prediction Error Around First Anomaly")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 12. Compare errors before, inside and after first anomaly
# --------------------------------------------------

inside = local_errors[
    (local_times >= 2149) &
    (local_times <= 2349)
]

before = local_errors[local_times < 2149]
after = local_errors[local_times > 2349]

print("\n--- Local Error Comparison ---")

print("Before first anomaly:", np.mean(before))
print("Inside first anomaly:", np.mean(inside))
print("After first anomaly:", np.mean(after))


# --------------------------------------------------
# 13. Save trained model
# --------------------------------------------------

model.save("p1_lstm_forecaster.keras")

print("\nSaved model: p1_lstm_forecaster.keras")
