import numpy as np

from sklearn.ensemble import IsolationForest


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

TRAIN_PATH = "../data/train/G-7.npy"
TEST_PATH = "../data/test/G-7.npy"

WINDOW_SIZE = 10
RANDOM_STATE = 42

ANOMALIES = [
    (3650, 3750, "A1 Contextual"),
    (5050, 5100, "A2 Point"),
    (7560, 7675, "A3 Contextual")
]

THRESHOLD_PERCENTILE = 95

MAX_GAP = 2
MIN_EVENT_LENGTH = 3

IOU_THRESHOLD = 0.1


# --------------------------------------------------
# 2. Load data
# --------------------------------------------------

train_data = np.load(TRAIN_PATH).astype(np.float32)
test_data = np.load(TEST_PATH).astype(np.float32)

train = train_data[:, 0]
test = test_data[:, 0]


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


# --------------------------------------------------
# 4. Training / validation split
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

validation_scores = -model.score_samples(
    X_validation
)

test_scores = -model.score_samples(X_test)


# --------------------------------------------------
# 7. Apply threshold
# --------------------------------------------------

threshold = np.percentile(
    validation_scores,
    THRESHOLD_PERCENTILE
)

predictions = (
    test_scores > threshold
).astype(int)


# --------------------------------------------------
# 8. Temporal grouping
# --------------------------------------------------

def fill_short_gaps(predictions, max_gap):

    result = predictions.copy()

    indices = np.where(result == 1)[0]

    for i in range(len(indices) - 1):

        current = indices[i]
        following = indices[i + 1]

        gap = following - current - 1

        if 0 < gap <= max_gap:

            result[current + 1:following] = 1

    return result


def remove_short_events(predictions, min_length):

    result = predictions.copy()

    indices = np.where(result == 1)[0]

    if len(indices) == 0:
        return result

    runs = []

    start = indices[0]
    previous = indices[0]

    for index in indices[1:]:

        if index == previous + 1:
            previous = index

        else:
            runs.append((start, previous))
            start = index
            previous = index

    runs.append((start, previous))

    for start, end in runs:

        if end - start + 1 < min_length:

            result[start:end + 1] = 0

    return result


predictions = fill_short_gaps(
    predictions,
    MAX_GAP
)

predictions = remove_short_events(
    predictions,
    MIN_EVENT_LENGTH
)


# --------------------------------------------------
# 9. Extract predicted events
# --------------------------------------------------

def extract_events(predictions, target_timesteps):

    indices = np.where(predictions == 1)[0]

    if len(indices) == 0:
        return []

    events = []

    start = indices[0]
    previous = indices[0]

    for index in indices[1:]:

        if index == previous + 1:
            previous = index

        else:

            events.append((
                int(target_timesteps[start]),
                int(target_timesteps[previous])
            ))

            start = index
            previous = index

    events.append((
        int(target_timesteps[start]),
        int(target_timesteps[previous])
    ))

    return events


target_timesteps = np.arange(
    WINDOW_SIZE,
    len(test)
)

predicted_events = extract_events(
    predictions,
    target_timesteps
)


# --------------------------------------------------
# 10. IoU calculation
# --------------------------------------------------

def calculate_iou(event_a, event_b):

    start_a, end_a = event_a
    start_b, end_b = event_b

    intersection = max(
        0,
        min(end_a, end_b) - max(start_a, start_b) + 1
    )

    union = (
        (end_a - start_a + 1)
        + (end_b - start_b + 1)
        - intersection
    )

    return intersection / union if union else 0


# --------------------------------------------------
# 11. Event matching
# --------------------------------------------------

print("\nGROUND-TRUTH EVENTS")
print("=" * 55)

for start, end, name in ANOMALIES:

    print(f"{name}: {start} - {end}")


print("\nPREDICTED EVENTS")
print("=" * 55)

for i, event in enumerate(predicted_events):

    print(
        f"Event {i + 1}: "
        f"{event[0]} - {event[1]}"
    )


print("\nEVENT MATCHING")
print("=" * 55)

matched_predictions = set()
matched_ground_truth = set()

matches = []

for gt_index, (gt_start, gt_end, name) in enumerate(ANOMALIES):

    best_iou = 0
    best_prediction = None

    for pred_index, predicted_event in enumerate(predicted_events):

        iou = calculate_iou(
            (gt_start, gt_end),
            predicted_event
        )

        if iou > best_iou:

            best_iou = iou
            best_prediction = pred_index

    print(f"\n{name}")

    if best_prediction is not None:

        predicted_event = predicted_events[best_prediction]

        print("Best predicted event:", predicted_event)
        print(f"IoU: {best_iou:.4f}")

        if best_iou >= IOU_THRESHOLD:

            matches.append((gt_index, best_prediction, best_iou))

            matched_ground_truth.add(gt_index)
            matched_predictions.add(best_prediction)

            print("Status: MATCHED")

        else:

            print("Status: NOT MATCHED")

    else:

        print("Status: NO PREDICTED EVENT")


# --------------------------------------------------
# 12. Event-level metrics
# --------------------------------------------------

event_tp = len(matched_ground_truth)

event_fp = len(predicted_events) - len(matched_predictions)

event_fn = len(ANOMALIES) - event_tp

event_precision = (
    event_tp / (event_tp + event_fp)
    if event_tp + event_fp else 0
)

event_recall = (
    event_tp / (event_tp + event_fn)
    if event_tp + event_fn else 0
)

event_f1 = (
    2 * event_precision * event_recall /
    (event_precision + event_recall)
    if event_precision + event_recall else 0
)


print("\nEVENT-LEVEL RESULTS")
print("=" * 55)

print("Ground-truth events:", len(ANOMALIES))
print("Predicted events:", len(predicted_events))
print("Matched events:", event_tp)

print("Event false positives:", event_fp)
print("Event false negatives:", event_fn)

print(f"Event precision: {event_precision:.4f}")
print(f"Event recall: {event_recall:.4f}")
print(f"Event F1-score: {event_f1:.4f}")