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

TRAIN_PATH = "../data/train/G-7.npy"
TEST_PATH = "../data/test/G-7.npy"

WINDOW_SIZE = 10
RANDOM_STATE = 42
THRESHOLD_PERCENTILE = 99

ANOMALIES = [
    (3650, 3750, "A1 Contextual"),
    (5050, 5100, "A2 Point"),
    (7560, 7675, "A3 Contextual")
]

# --------------------------------------------------
# 2. Load data
# --------------------------------------------------

train_data = np.load(TRAIN_PATH).astype(np.float32)
test_data = np.load(TEST_PATH).astype(np.float32)

train = train_data[:, 0]
test = test_data[:, 0]

print("Training shape:", train_data.shape)
print("Testing shape:", test_data.shape)

# --------------------------------------------------
# 3. Feature extraction
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

        frequency_minus_02 = np.mean(
            np.isclose(window, -0.2, atol=1e-6)
        )

        frequency_minus_1 = np.mean(
            np.isclose(window, -1.0, atol=1e-6)
        )

        # Count consecutive -1.0 observations ending
        # at the previous timestep.
        run_length = 0

        for value in window[::-1]:

            if np.isclose(value, -1.0, atol=1e-6):
                run_length += 1
            else:
                break

        features.append([
            current,
            previous,
            absolute_difference,
            rolling_mean,
            rolling_std,
            frequency_minus_02,
            frequency_minus_1,
            run_length
        ])

    return np.array(features, dtype=np.float32)


X_train = create_features(train, WINDOW_SIZE)
X_test = create_features(test, WINDOW_SIZE)

print("\nFeature shapes:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

# --------------------------------------------------
# 4. Chronological training / validation split
# --------------------------------------------------

split_idx = int(len(train) * 0.8)

fit_data = train[:split_idx]

validation_data = train[
    split_idx - WINDOW_SIZE:
]

X_fit = create_features(fit_data, WINDOW_SIZE)

X_validation = create_features(
    validation_data,
    WINDOW_SIZE
)

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

model.fit(X_fit)

# --------------------------------------------------
# 6. Calculate anomaly scores
# --------------------------------------------------

# Higher score = more anomalous

train_scores = -model.score_samples(X_fit)

validation_scores = -model.score_samples(
    X_validation
)

test_scores = -model.score_samples(X_test)


# --------------------------------------------------
# 8. Create ground truth
# --------------------------------------------------

target_timesteps = np.arange(
    WINDOW_SIZE,
    len(test)
)

ground_truth = np.zeros(
    len(target_timesteps),
    dtype=int
)

for start, end, name in ANOMALIES:

    ground_truth[
        (target_timesteps >= start) &
        (target_timesteps <= end)
    ] = 1

# --------------------------------------------------
# 7. Threshold sensitivity analysis
# --------------------------------------------------

percentiles = [95, 97, 98, 99]

print("\nTHRESHOLD SENSITIVITY ANALYSIS")
print("=" * 75)

threshold_results = {}

for percentile in percentiles:

    threshold = np.percentile(
        validation_scores,
        percentile
    )

    predictions = (
        test_scores > threshold
    ).astype(int)

    threshold_results[percentile] = (
        threshold,
        predictions
    )

    tn, fp, fn, tp = confusion_matrix(
        ground_truth,
        predictions,
        labels=[0, 1]
    ).ravel()

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    print(f"\nValidation percentile: {percentile}")
    print(f"Threshold: {threshold:.6f}")
    print(f"True positives: {tp}")
    print(f"False positives: {fp}")
    print(f"False negatives: {fn}")
    print(f"True negatives: {tn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    for start, end, name in ANOMALIES:

        mask = (
            (target_timesteps >= start) &
            (target_timesteps <= end)
        )

        detected = np.sum(predictions[mask])
        total = np.sum(mask)

        print(
            f"{name} recall: "
            f"{detected}/{total} "
            f"({detected / total:.4f})"
        )

# Keep 99th percentile as the original baseline
threshold, predictions = threshold_results[95]

print("\nOriginal baseline retained: 99th percentile")

# --------------------------------------------------
# 9. Temporal grouping experiment
# --------------------------------------------------

def fill_short_gaps(predictions, max_gap):

    result = predictions.copy()

    if max_gap == 0:
        return result

    positive_indices = np.where(result == 1)[0]

    for i in range(len(positive_indices) - 1):

        current = positive_indices[i]
        following = positive_indices[i + 1]

        gap = following - current - 1

        if 0 < gap <= max_gap:

            result[current + 1:following] = 1

    return result


def remove_short_events(predictions, min_length):

    result = predictions.copy()

    if min_length <= 1:
        return result

    indices = np.where(result == 1)[0]

    if len(indices) == 0:
        return result

    run_start = indices[0]
    previous = indices[0]

    runs = []

    for index in indices[1:]:

        if index == previous + 1:
            previous = index

        else:
            runs.append((run_start, previous))

            run_start = index
            previous = index

    runs.append((run_start, previous))

    for start, end in runs:

        run_length = end - start + 1

        if run_length < min_length:

            result[start:end + 1] = 0

    return result


def temporal_grouping(predictions, max_gap, min_length):

    result = fill_short_gaps(
        predictions,
        max_gap
    )

    result = remove_short_events(
        result,
        min_length
    )

    return result


# --------------------------------------------------
# 10. Compare temporal grouping configurations
# --------------------------------------------------

configurations = [
    (0, 1),
    (1, 1),
    (2, 1),
    (2, 2),
    (2, 3),
    (3, 3)
]

print("\nTEMPORAL GROUPING COMPARISON")
print("=" * 75)

grouping_results = {}

for max_gap, min_length in configurations:

    grouped = temporal_grouping(
        predictions,
        max_gap,
        min_length
    )

    grouping_results[(max_gap, min_length)] = grouped

    tn, fp, fn, tp = confusion_matrix(
        ground_truth,
        grouped,
        labels=[0, 1]
    ).ravel()

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    print(f"\nMaximum gap: {max_gap}")
    print(f"Minimum event length: {min_length}")

    print("True positives:", tp)
    print("False positives:", fp)
    print("False negatives:", fn)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    for start, end, name in ANOMALIES:

        mask = (
            (target_timesteps >= start) &
            (target_timesteps <= end)
        )

        detected = np.sum(grouped[mask])
        total = np.sum(mask)

        print(
            f"{name}: {detected}/{total} "
            f"({detected / total:.4f})"
        )

# --------------------------------------------------
# 11. Count detected events
# --------------------------------------------------

def count_events(predictions):

    indices = np.where(predictions == 1)[0]

    if len(indices) == 0:
        return 0

    event_count = 1

    for i in range(1, len(indices)):

        if indices[i] != indices[i - 1] + 1:
            event_count += 1

    return event_count


print("\nDETECTED EVENT COUNTS")

for config, grouped in grouping_results.items():

    print(
        f"Gap={config[0]}, "
        f"Min length={config[1]}: "
        f"{count_events(grouped)} events"
    )

print(
    "\nRaw 95th-percentile events:",
    count_events(predictions)
)


# --------------------------------------------------
# 9. Evaluation function
# --------------------------------------------------



def evaluate(name, y_true, y_pred, scores):

    print("\n" + "=" * 55)
    print(name)
    print("=" * 55)

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


# Overall evaluation
evaluate(
    "G-7 Overall",
    ground_truth,
    predictions,
    test_scores
)

# Individual anomaly evaluations
for start, end, name in ANOMALIES:

    mask = (
        (target_timesteps >= start - 100) &
        (target_timesteps <= end + 100)
    )

    local_truth = (
        (target_timesteps[mask] >= start) &
        (target_timesteps[mask] <= end)
    ).astype(int)

    local_predictions = predictions[mask]
    local_scores = test_scores[mask]

    evaluate(
        name,
        local_truth,
        local_predictions,
        local_scores
    )

# --------------------------------------------------
# 10. Anomaly score statistics
# --------------------------------------------------

print("\nIndividual Anomaly Score Statistics")

for start, end, name in ANOMALIES:

    mask = (
        (target_timesteps >= start) &
        (target_timesteps <= end)
    )

    scores = test_scores[mask]

    print(f"\n{name}")
    print("Mean:", np.mean(scores))
    print("Median:", np.median(scores))
    print("Maximum:", np.max(scores))

    print(
        "Detected:",
        np.sum(predictions[mask]),
        "/",
        len(scores)
    )

# --------------------------------------------------
# 11. Plot all anomaly scores
# --------------------------------------------------

plt.figure(figsize=(16, 6))

plt.plot(
    target_timesteps,
    test_scores,
    linewidth=0.7,
    label="Isolation Forest anomaly score"
)

plt.axhline(
    threshold,
    color="red",
    linestyle="--",
    label=f"Threshold = {threshold:.4f}"
)

for start, end, name in ANOMALIES:

    plt.axvspan(
        start,
        end,
        alpha=0.2,
        label=name
    )

plt.title("G-7 Isolation Forest Anomaly Detection")
plt.xlabel("Timestep")
plt.ylabel("Anomaly score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 12. Plot individual anomaly regions
# --------------------------------------------------

for start, end, name in ANOMALIES:

    mask = (
        (target_timesteps >= start - 50) &
        (target_timesteps <= end + 50)
    )

    plt.figure(figsize=(14, 4))

    plt.plot(
        target_timesteps[mask],
        test_scores[mask],
        label="Anomaly score"
    )

    plt.axhline(
        threshold,
        color="red",
        linestyle="--",
        label="Threshold"
    )

    plt.axvspan(
        start,
        end,
        color="orange",
        alpha=0.25,
        label="Ground truth"
    )

    plt.title(f"G-7: {name}")
    plt.xlabel("Timestep")
    plt.ylabel("Anomaly score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --------------------------------------------------
# 13. Save model
# --------------------------------------------------

import joblib

joblib.dump(
    model,
    "g7_isolation_forest.joblib"
)

print("\nModel saved: g7_isolation_forest.joblib")