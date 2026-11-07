
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

TRAIN_PATH = "../data/train/G-7.npy"
TEST_PATH = "../data/test/G-7.npy"

WINDOW_SIZE = 10
RANDOM_STATE = 42

THRESHOLD_PERCENTILE = 95

# Compare different temporal grouping gaps
GAP_VALUES = [10]

MIN_EVENT_LENGTH = 3

IOU_THRESHOLDS = [0.1, 0.3, 0.5, 0.7]

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
# 7. Apply validation-calibrated threshold
# --------------------------------------------------

threshold = np.percentile(
    validation_scores,
    THRESHOLD_PERCENTILE
)

raw_predictions = (
    test_scores > threshold
).astype(int)

print("Threshold:", threshold)
print("Threshold percentile:", THRESHOLD_PERCENTILE)


# --------------------------------------------------
# 8. Temporal grouping functions
# --------------------------------------------------

def fill_short_gaps(predictions, max_gap):

    result = predictions.copy()

    if max_gap == 0:
        return result

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

        duration = end - start + 1

        if duration < min_length:
            result[start:end + 1] = 0

    return result


def apply_grouping(predictions, max_gap, min_length):

    grouped = fill_short_gaps(
        predictions,
        max_gap
    )

    grouped = remove_short_events(
        grouped,
        min_length
    )

    return grouped


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


# --------------------------------------------------
# 10. Ground-truth labels
# --------------------------------------------------

target_timesteps = np.arange(
    WINDOW_SIZE,
    len(test)
)

y_true = np.zeros(len(target_timesteps), dtype=int)

for start, end, name in ANOMALIES:

    mask = (
        (target_timesteps >= start) &
        (target_timesteps <= end)
    )

    y_true[mask] = 1


# --------------------------------------------------
# 11. IoU calculation
# --------------------------------------------------

def calculate_iou(event_a, event_b):

    start_a, end_a = event_a
    start_b, end_b = event_b

    intersection = max(
        0,
        min(end_a, end_b) - max(start_a, start_b) + 1
    )

    length_a = end_a - start_a + 1
    length_b = end_b - start_b + 1

    union = length_a + length_b - intersection

    return intersection / union if union else 0


# --------------------------------------------------
# 12. One-to-one event matching
# --------------------------------------------------

def match_events(ground_truth, predicted_events, iou_threshold):

    candidate_pairs = []

    for gt_index, (gt_start, gt_end, name) in enumerate(ANOMALIES):

        for pred_index, predicted_event in enumerate(predicted_events):

            iou = calculate_iou(
                (gt_start, gt_end),
                predicted_event
            )

            if iou >= iou_threshold:

                candidate_pairs.append((
                    iou,
                    gt_index,
                    pred_index
                ))

    candidate_pairs.sort(
        key=lambda item: item[0],
        reverse=True
    )

    matched_gt = set()
    matched_pred = set()
    matches = []

    for iou, gt_index, pred_index in candidate_pairs:

        if (
            gt_index not in matched_gt
            and pred_index not in matched_pred
        ):

            matched_gt.add(gt_index)
            matched_pred.add(pred_index)

            matches.append((
                gt_index,
                pred_index,
                iou
            ))

    event_tp = len(matches)
    event_fp = len(predicted_events) - len(matched_pred)
    event_fn = len(ANOMALIES) - len(matched_gt)

    return event_tp, event_fp, event_fn, matches

# --------------------------------------------------
# 13. Evaluate each grouping configuration
# --------------------------------------------------

print("\nG-7 EVENT GROUPING EXPERIMENT")
print("=" * 90)

results = []

for max_gap in GAP_VALUES:

    predictions = apply_grouping(
        raw_predictions,
        max_gap,
        MIN_EVENT_LENGTH
    )

    # Sample-level metrics
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    # Extract predicted intervals
    predicted_events = extract_events(
        predictions,
        target_timesteps
    )

    
    # Event-level IoU sensitivity analysis

predicted_events = extract_events(
    predictions,
    target_timesteps
)

print("\nEvent-level IoU sensitivity:")

for iou_threshold in IOU_THRESHOLDS:

    event_tp, event_fp, event_fn, matches = match_events(
        ANOMALIES,
        predicted_events,
        iou_threshold
    )

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

    print(f"\nIoU threshold: {iou_threshold}")

    print("Matched events:", event_tp)
    print("Event FP:", event_fp)
    print("Event FN:", event_fn)

    print(f"Event precision: {event_precision:.4f}")
    print(f"Event recall:    {event_recall:.4f}")
    print(f"Event F1-score:  {event_f1:.4f}")

    for gt_index, pred_index, iou in matches:

        print(
            f"{ANOMALIES[gt_index][2]} "
            f"<-> {predicted_events[pred_index]} "
            f"| IoU: {iou:.4f}"
        )

    # Print configuration results
    print(f"\nMAX GAP = {max_gap}")
    print("-" * 90)

    print("Sample-level results:")
    print(f"TP: {tp} | FP: {fp} | FN: {fn} | TN: {tn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\nEvent-level results:")
    print("Predicted events:", len(predicted_events))
    print("Matched events:", event_tp)
    print("Event FP:", event_fp)
    print("Event FN:", event_fn)

    print(f"Event precision: {event_precision:.4f}")
    print(f"Event recall:    {event_recall:.4f}")
    print(f"Event F1-score:  {event_f1:.4f}")

    print("\nEvent matches:")

    for gt_index, pred_index, iou in matches:

        print(
            f"{ANOMALIES[gt_index][2]} "
            f"<-> {predicted_events[pred_index]} "
            f"| IoU: {iou:.4f}"
        )

    print("\nPredicted intervals:")

    for i, event in enumerate(predicted_events):

        print(f"Event {i + 1}: {event[0]} - {event[1]}")


# --------------------------------------------------
# 14. Comparison summary
# --------------------------------------------------

print("\n\nFINAL COMPARISON")
print("=" * 110)

print(
    f"{'Gap':<8}"
    f"{'TP':<7}"
    f"{'FP':<7}"
    f"{'FN':<7}"
    f"{'Sample F1':<13}"
    f"{'Events':<10}"
    f"{'Event TP':<11}"
    f"{'Event F1':<10}"
)

for result in results:

    print(
        f"{result['gap']:<8}"
        f"{result['tp']:<7}"
        f"{result['fp']:<7}"
        f"{result['fn']:<7}"
        f"{result['f1']:<13.4f}"
        f"{result['predicted_events']:<10}"
        f"{result['event_tp']:<11}"
        f"{result['event_f1']:<10.4f}"
    )
