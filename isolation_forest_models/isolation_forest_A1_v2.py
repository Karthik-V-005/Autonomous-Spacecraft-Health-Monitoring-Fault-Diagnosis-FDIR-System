import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score
)

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

TRAIN_PATH = "../data/train/A-1.npy"
TEST_PATH = "../data/test/A-1.npy"

WINDOW_SIZE = 10
RANDOM_STATE = 42

ANOMALY_START = 4690
ANOMALY_END = 4774
POINT_ANOMALY = 4750

# --------------------------------------------------
# 2. Load data
# --------------------------------------------------

train_data = np.load(TRAIN_PATH).astype(np.float32)
test_data = np.load(TEST_PATH).astype(np.float32)

train_telemetry = train_data[:, 0]
test_telemetry = test_data[:, 0]

print("Training shape:", train_data.shape)
print("Testing shape:", test_data.shape)

# --------------------------------------------------
# 3. Create telemetry-based features
# --------------------------------------------------

def create_features(telemetry, window_size):

    features = []

    for t in range(window_size, len(telemetry)):

        current = telemetry[t]
        previous = telemetry[t - 1]

        window = telemetry[t - window_size:t]

        rolling_mean = np.mean(window)
        rolling_std = np.std(window)

        absolute_difference = abs(current - previous)

        deviation = abs(current - rolling_mean)

        features.append([
            current,
            previous,
            absolute_difference,
            rolling_mean,
            rolling_std,
            deviation
        ])

    return np.array(features, dtype=np.float32)


X_train = create_features(train_telemetry, WINDOW_SIZE)
X_test = create_features(test_telemetry, WINDOW_SIZE)

print("\nFeature shapes:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

# --------------------------------------------------
# 4. Train Isolation Forest
# --------------------------------------------------

model = IsolationForest(
    n_estimators=200,
    max_samples="auto",
    contamination="auto",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model.fit(X_train)

# --------------------------------------------------
# 5. Calculate anomaly scores
# --------------------------------------------------

# Higher score = more anomalous
train_scores = -model.score_samples(X_train)
test_scores = -model.score_samples(X_test)

# --------------------------------------------------
# 6. Validation threshold
# --------------------------------------------------

# The training data is constant, so use its final
# portion as normal validation data.

split_idx = int(len(train_telemetry) * 0.8)

validation_data = train_telemetry[
    split_idx - WINDOW_SIZE:
]

X_validation = create_features(
    validation_data,
    WINDOW_SIZE
)

validation_scores = -model.score_samples(X_validation)

threshold = np.max(validation_scores)

print("\nValidation threshold:", threshold)

# Strictly greater than threshold
predictions = (test_scores > threshold).astype(int)

# --------------------------------------------------
# 7. Create ground-truth labels
# --------------------------------------------------

target_timesteps = np.arange(
    WINDOW_SIZE,
    len(test_telemetry)
)

# Original 85-timestep annotation
ground_truth_interval = (
    (target_timesteps >= ANOMALY_START) &
    (target_timesteps <= ANOMALY_END)
).astype(int)

# Single unusual measurement
ground_truth_point = (
    target_timesteps == POINT_ANOMALY
).astype(int)

# --------------------------------------------------
# 8. Evaluation function
# --------------------------------------------------

def evaluate_model(name, y_true, y_pred, scores):

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=["Normal", "Anomaly"],
        zero_division=0
    ))

    print("ROC-AUC:", roc_auc_score(y_true, scores))

    print(
        "Average Precision:",
        average_precision_score(y_true, scores)
    )


# Evaluate the original 85-sample annotation
evaluate_model(
    "Original Ground Truth: 85 Samples",
    ground_truth_interval,
    predictions,
    test_scores
)

# Evaluate the single unusual measurement
evaluate_model(
    "Observed Point Deviation: Timestep 4750",
    ground_truth_point,
    predictions,
    test_scores
)

# --------------------------------------------------
# 9. Inspect important anomaly scores
# --------------------------------------------------

print("\nImportant Timesteps")
print("-" * 40)

for timestep in [4749, 4750, 4751, 4752]:

    index = timestep - WINDOW_SIZE

    print(
        f"Timestep: {timestep}, "
        f"Telemetry: {test_telemetry[timestep]}, "
        f"Anomaly score: {test_scores[index]:.6f}, "
        f"Predicted: {predictions[index]}"
    )

# --------------------------------------------------
# 10. Plot anomaly scores
# --------------------------------------------------

plt.figure(figsize=(15, 5))

plt.plot(
    target_timesteps,
    test_scores,
    label="Isolation Forest anomaly score",
    linewidth=0.8
)

plt.axhline(
    threshold,
    color="red",
    linestyle="--",
    label=f"Threshold = {threshold:.4f}"
)

plt.axvspan(
    ANOMALY_START,
    ANOMALY_END,
    color="orange",
    alpha=0.2,
    label="Original ground-truth interval"
)

plt.axvline(
    POINT_ANOMALY,
    color="purple",
    linestyle=":",
    label="Unusual measurement at 4750"
)

plt.xlim(4650, 4800)

plt.title("A-1 Isolation Forest v2")
plt.xlabel("Timestep")
plt.ylabel("Anomaly score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 11. Compare feature values around anomaly
# --------------------------------------------------

print("\nFeature Comparison Around Timestep 4750")
print("-" * 50)

feature_names = [
    "Current telemetry",
    "Previous telemetry",
    "Absolute difference",
    "Rolling mean",
    "Rolling standard deviation",
    "Deviation from rolling mean"
]

for timestep in [4749, 4750, 4751, 4752]:

    index = timestep - WINDOW_SIZE

    print(f"\nTimestep {timestep}:")

    for name, value in zip(
        feature_names,
        X_test[index]
    ):
        print(f"{name}: {value:.6f}")