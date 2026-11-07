
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

ANOMALY_SEQUENCES = [
    (2149, 2349),
    (3539, 3779),
    (4536, 4844)
]


# --------------------------------------------------
# 2. Load all telemetry and command features
# --------------------------------------------------

train_data = np.load(TRAIN_PATH).astype(np.float32)
test_data = np.load(TEST_PATH).astype(np.float32)

print("Training data shape:", train_data.shape)
print("Test data shape:", test_data.shape)

# Column 0 is the telemetry value.
# Remaining columns are command-related features.
NUM_FEATURES = train_data.shape[1]

print("Number of input features:", NUM_FEATURES)


# --------------------------------------------------
# 3. Create multivariate sliding windows
# --------------------------------------------------

def create_windows(data, window_size):
    X, y = [], []

    for i in range(len(data) - window_size):
        # Use all features from the previous timesteps.
        X.append(data[i:i + window_size])

        # Predict only the next telemetry value (column 0).
        y.append(data[i + window_size, 0])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    return X, y


# Chronological 80/20 training-validation split
split_idx = int(len(train_data) * 0.8)

fit_data = train_data[:split_idx]

# Preserve the preceding window as validation context.
val_data = train_data[split_idx - WINDOW_SIZE:]

X_train, y_train = create_windows(fit_data, WINDOW_SIZE)
X_val, y_val = create_windows(val_data, WINDOW_SIZE)
X_test, y_test = create_windows(test_data, WINDOW_SIZE)

print("\nWindow shapes:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_val:", X_val.shape)
print("y_val:", y_val.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# --------------------------------------------------
# 4. Build and train the multivariate LSTM
# --------------------------------------------------

model = tf.keras.Sequential([
    tf.keras.layers.Input(
        shape=(WINDOW_SIZE, NUM_FEATURES)
    ),
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

print("\n--- Forecasting Results ---")

print("Test MAE:", np.mean(errors))
print("Maximum test error:", np.max(errors))

print("\nValidation error percentiles:")
print(np.percentile(val_errors, [50, 90, 95, 99, 99.5, 100]))

print("\nTest error percentiles:")
print(np.percentile(errors, [50, 90, 95, 99, 99.5, 100]))

print("\nValidation mean error:", np.mean(val_errors))
print("Test mean error:", np.mean(errors))


# --------------------------------------------------
# 6. Calibrate anomaly threshold
# --------------------------------------------------

threshold = np.percentile(
    val_errors,
    THRESHOLD_PERCENTILE
)

print(
    f"\n{THRESHOLD_PERCENTILE}th-percentile threshold:",
    threshold
)


# --------------------------------------------------
# 7. Construct ground-truth labels
# --------------------------------------------------

target_timesteps = np.arange(
    WINDOW_SIZE,
    len(test_data)
)

ground_truth = np.zeros(len(target_timesteps), dtype=int)

for start, end in ANOMALY_SEQUENCES:
    mask = (
        (target_timesteps >= start) &
        (target_timesteps <= end)
    )
    ground_truth[mask] = 1


# --------------------------------------------------
# 8. Evaluate anomaly detection
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

print("\n--- Multivariate LSTM Ranking Evaluation ---")
print("ROC-AUC:", round(roc_auc, 4))
print("Average Precision:", round(average_precision, 4))



# --------------------------------------------------
# 9. Compare normal and anomalous errors
# --------------------------------------------------

normal_errors = errors[ground_truth == 0]
anomaly_errors = errors[ground_truth == 1]

print("\n--- Overall Error Comparison ---")
print("Mean error outside anomalies:", np.mean(normal_errors))
print("Mean error inside anomalies:", np.mean(anomaly_errors))
print("Maximum error outside anomalies:", np.max(normal_errors))
print("Maximum error inside anomalies:", np.max(anomaly_errors))

# --------------------------------------------------
# 10. Compare prediction errors across anomaly intervals
# --------------------------------------------------

print("\n--- Individual Anomaly Error Analysis ---")

for i, (start, end) in enumerate(ANOMALY_SEQUENCES, start=1):

    mask = (
        (target_timesteps >= start) &
        (target_timesteps <= end)
    )

    interval_errors = errors[mask]

    print(f"\nAnomaly {i}: {start} - {end}")
    print("Mean error:", np.mean(interval_errors))
    print("Maximum error:", np.max(interval_errors))
    print("Median error:", np.median(interval_errors))
    print("Errors above threshold:",
          np.sum(interval_errors > threshold))
    print("Total anomalous timesteps:", len(interval_errors))



# --------------------------------------------------
# 10. Analyze command features
# --------------------------------------------------

print("\n--- Command Feature Analysis ---")

# Remove the first WINDOW_SIZE rows to align
# command features with prediction target timesteps.
command_data = test_data[WINDOW_SIZE:, 1:]

print("Aligned command data shape:", command_data.shape)
print("Ground-truth shape:", ground_truth.shape)

# Calculate percentage of active timesteps
command_activity = np.mean(command_data != 0, axis=0) * 100

for i, activity in enumerate(command_activity, start=1):
    print(f"Command feature {i}: {activity:.2f}% active")


# Separate normal and anomalous command data
normal_command_data = command_data[ground_truth == 0]
anomaly_command_data = command_data[ground_truth == 1]

normal_activity = np.mean(
    normal_command_data != 0, axis=0
) * 100

anomaly_activity = np.mean(
    anomaly_command_data != 0, axis=0
) * 100

print("\n--- Command Activity Comparison ---")

for i in range(NUM_FEATURES - 1):
    print(
        f"Feature {i + 1}: "
        f"Normal = {normal_activity[i]:.2f}%, "
        f"Anomaly = {anomaly_activity[i]:.2f}%"
    )


# Plot command activity comparison
features = np.arange(1, NUM_FEATURES)

plt.figure(figsize=(14, 5))

plt.bar(
    features - 0.2,
    normal_activity,
    width=0.4,
    label="Normal"
)

plt.bar(
    features + 0.2,
    anomaly_activity,
    width=0.4,
    label="Anomalous"
)

plt.xlabel("Command Feature")
plt.ylabel("Active Timesteps (%)")
plt.title("P-1: Command Feature Activity")
plt.xticks(features)
plt.legend()
plt.grid(axis="y")
plt.tight_layout()
plt.show()

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
plt.title("P-1 Multivariate LSTM Validation Error")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 11. Plot prediction errors around first anomaly
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
    label="Multivariate LSTM prediction error",
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
plt.title("P-1: Multivariate LSTM Error Around First Anomaly")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 12. Save the trained model
# --------------------------------------------------

model.save("p1_multivariate_lstm_forecaster.keras")

print("\nSaved model: p1_multivariate_lstm_forecaster.keras")
