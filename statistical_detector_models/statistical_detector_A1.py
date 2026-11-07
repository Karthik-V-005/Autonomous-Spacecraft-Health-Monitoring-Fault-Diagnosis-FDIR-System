import numpy as np
import matplotlib.pyplot as plt

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

THRESHOLD = 0.01

ANOMALY_START = 4690
ANOMALY_END = 4774
POINT_ANOMALY = 4750

# --------------------------------------------------
# 2. Load data
# --------------------------------------------------

train_data = np.load(TRAIN_PATH)
test_data = np.load(TEST_PATH)

train_telemetry = train_data[:, 0]
test_telemetry = test_data[:, 0]

print("Training samples:", len(train_telemetry))
print("Testing samples:", len(test_telemetry))

# --------------------------------------------------
# 3. Learn normal behaviour
# --------------------------------------------------

normal_mean = np.mean(train_telemetry)
normal_std = np.std(train_telemetry)

print("\nTraining Statistics")
print("-------------------")
print("Mean:", normal_mean)
print("Standard deviation:", normal_std)

# --------------------------------------------------
# 4. Calculate anomaly scores
# --------------------------------------------------

anomaly_scores = np.abs(
    test_telemetry - normal_mean
)

predictions = (
    anomaly_scores > THRESHOLD
).astype(int)

print("\nThreshold:", THRESHOLD)

# --------------------------------------------------
# 5. Create ground-truth labels
# --------------------------------------------------

timesteps = np.arange(len(test_telemetry))

# Original 85-sample annotation
ground_truth_interval = (
    (timesteps >= ANOMALY_START) &
    (timesteps <= ANOMALY_END)
).astype(int)

# Single unusual measurement
ground_truth_point = (
    timesteps == POINT_ANOMALY
).astype(int)

# --------------------------------------------------
# 6. Evaluation
# --------------------------------------------------

def evaluate(name, y_true, y_pred, scores):

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


evaluate(
    "Original Ground Truth: 85 Samples",
    ground_truth_interval,
    predictions,
    anomaly_scores
)

evaluate(
    "Observed Point Deviation: Timestep 4750",
    ground_truth_point,
    predictions,
    anomaly_scores
)

# --------------------------------------------------
# 7. Inspect important timesteps
# --------------------------------------------------

print("\nImportant Timesteps")
print("-" * 40)

for t in [4749, 4750, 4751, 4752]:

    print(
        f"Timestep: {t}, "
        f"Value: {test_telemetry[t]}, "
        f"Score: {anomaly_scores[t]:.6f}, "
        f"Prediction: {predictions[t]}"
    )

# --------------------------------------------------
# 8. Plot anomaly scores
# --------------------------------------------------

plt.figure(figsize=(15, 5))

plt.plot(
    timesteps,
    anomaly_scores,
    label="Statistical anomaly score",
    linewidth=0.8
)

plt.axhline(
    THRESHOLD,
    color="red",
    linestyle="--",
    label=f"Threshold = {THRESHOLD}"
)

plt.axvspan(
    ANOMALY_START,
    ANOMALY_END,
    color="orange",
    alpha=0.2,
    label="Ground-truth interval"
)

plt.axvline(
    POINT_ANOMALY,
    color="purple",
    linestyle=":",
    label="Unusual measurement"
)

plt.xlim(4650, 4800)

plt.title("A-1 Statistical Anomaly Detection")
plt.xlabel("Timestep")
plt.ylabel("Absolute deviation from training mean")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()