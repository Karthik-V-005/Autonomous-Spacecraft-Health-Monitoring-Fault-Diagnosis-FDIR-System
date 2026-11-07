
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, classification_report


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

TRAIN_PATH = "data/train/P-1.npy"
TEST_PATH = "data/test/P-1.npy"

WINDOW_SIZE = 10
THRESHOLD_PERCENTILE = 99

ANOMALY_SEQUENCES = [
    (2149, 2349),
    (3539, 3779),
    (4536, 4844)
]

RANDOM_STATE = 42


# --------------------------------------------------
# 2. Load data
# --------------------------------------------------

train_data = np.load(TRAIN_PATH).astype(np.float32)
test_data = np.load(TEST_PATH).astype(np.float32)

print("Training data shape:", train_data.shape)
print("Test data shape:", test_data.shape)


# --------------------------------------------------
# 3. Construct features
# --------------------------------------------------

def create_features(data, window_size):
    telemetry = data[:, 0]
    commands = data[:, 1:]

    features = []

    for t in range(window_size, len(data)):

        # Historical telemetry observations
        history = telemetry[t - window_size:t]

        current_value = telemetry[t]
        previous_value = telemetry[t - 1]

        rolling_mean = np.mean(history)
        rolling_std = np.std(history)

        # Combine telemetry statistics and current commands
        feature_vector = np.concatenate([
            [
                current_value,
                previous_value,
                rolling_mean,
                rolling_std
            ],
            commands[t]
        ])

        features.append(feature_vector)

    return np.array(features, dtype=np.float32)


# --------------------------------------------------
# 4. Chronological training-validation split
# --------------------------------------------------

split_idx = int(len(train_data) * 0.8)

fit_data = train_data[:split_idx]

# Retain historical context for validation features
val_data = train_data[split_idx - WINDOW_SIZE:]

X_train = create_features(fit_data, WINDOW_SIZE)
X_val = create_features(val_data, WINDOW_SIZE)
X_test = create_features(test_data, WINDOW_SIZE)

print("\nFeature shapes:")
print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("X_test:", X_test.shape)

print("Number of features:", X_train.shape[1])


# --------------------------------------------------
# 5. Train Isolation Forest
# --------------------------------------------------

model = IsolationForest(
    n_estimators=200,
    max_samples="auto",
    contamination="auto",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model.fit(X_train)

print("\nIsolation Forest training completed.")


# --------------------------------------------------
# 6. Calculate anomaly scores
# --------------------------------------------------

# Isolation Forest gives lower scores to unusual observations.
# Negating the scores makes larger values indicate greater anomaly.

train_scores = -model.score_samples(X_train)
val_scores = -model.score_samples(X_val)
test_scores = -model.score_samples(X_test)

print("\n--- Anomaly Score Statistics ---")

print("Training mean score:", np.mean(train_scores))
print("Validation mean score:", np.mean(val_scores))
print("Test mean score:", np.mean(test_scores))


# --------------------------------------------------
# 7. Calibrate threshold using validation data
# --------------------------------------------------

threshold = np.percentile(
    val_scores,
    THRESHOLD_PERCENTILE
)

print(
    f"\n{THRESHOLD_PERCENTILE}th-percentile threshold:",
    threshold
)


# --------------------------------------------------
# 8. Construct ground-truth labels
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
# 9. Detect anomalies and evaluate
# --------------------------------------------------

predicted_labels = (test_scores > threshold).astype(int)

print("\n--- Isolation Forest Results ---")
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
# 10. Compare normal and anomalous scores
# --------------------------------------------------

normal_scores = test_scores[ground_truth == 0]
anomaly_scores = test_scores[ground_truth == 1]

print("\n--- Anomaly Score Comparison ---")

print("Mean normal score:", np.mean(normal_scores))
print("Mean anomaly score:", np.mean(anomaly_scores))

print("Maximum normal score:", np.max(normal_scores))
print("Maximum anomaly score:", np.max(anomaly_scores))



# --------------------------------------------------
# 10A. Threshold-independent evaluation
# --------------------------------------------------

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve
)

roc_auc = roc_auc_score(ground_truth, test_scores)

average_precision = average_precision_score(
    ground_truth,
    test_scores
)

print("\n--- Threshold-independent Evaluation ---")
print("ROC-AUC:", round(roc_auc, 4))
print("Average Precision:", round(average_precision, 4))


# ROC curve
fpr, tpr, _ = roc_curve(ground_truth, test_scores)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"Isolation Forest (AUC = {roc_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random ranking"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("P-1: Isolation Forest ROC Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Precision-recall curve
precision_values, recall_values, _ = precision_recall_curve(
    ground_truth,
    test_scores
)

plt.figure(figsize=(7, 5))

plt.plot(
    recall_values,
    precision_values,
    label=f"Average Precision = {average_precision:.3f}"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("P-1: Isolation Forest Precision-Recall Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 11. Analyze individual anomaly intervals
# --------------------------------------------------

print("\n--- Individual Anomaly Analysis ---")

for i, (start, end) in enumerate(ANOMALY_SEQUENCES, start=1):

    mask = (
        (target_timesteps >= start) &
        (target_timesteps <= end)
    )

    interval_scores = test_scores[mask]

    print(f"\nAnomaly {i}: {start} - {end}")
    print("Mean score:", np.mean(interval_scores))
    print("Median score:", np.median(interval_scores))
    print("Maximum score:", np.max(interval_scores))

    print(
        "Detected timesteps:",
        np.sum(interval_scores > threshold)
    )

    print("Total timesteps:", len(interval_scores))


# --------------------------------------------------
# 12. Plot validation score distribution
# --------------------------------------------------

plt.figure(figsize=(10, 4))

plt.hist(val_scores, bins=60)

plt.axvline(
    threshold,
    color="red",
    linestyle="--",
    label=f"Threshold = {threshold:.4f}"
)

plt.xlabel("Anomaly score")
plt.ylabel("Frequency")
plt.title("P-1 Isolation Forest Validation Scores")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 13. Plot anomaly scores around first anomaly
# --------------------------------------------------

start = 2100
end = 2400

local_mask = (
    (target_timesteps >= start) &
    (target_timesteps <= end)
)

local_times = target_timesteps[local_mask]
local_scores = test_scores[local_mask]

plt.figure(figsize=(14, 5))

plt.plot(
    local_times,
    local_scores,
    label="Isolation Forest anomaly score",
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
    label=f"Threshold = {threshold:.4f}"
)

plt.xlabel("Timestep")
plt.ylabel("Anomaly score")
plt.title("P-1: Isolation Forest Score Around First Anomaly")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 14. Save trained model
# --------------------------------------------------

joblib.dump(model, "p1_isolation_forest.joblib")

print("\nSaved model: p1_isolation_forest.joblib")
