
import ast
import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

from ashm_fdir.data_loader import load_all_channels
from ashm_fdir.detectors.temporal_detector import (
    TemporalAnomalyDetector
)


# --------------------------------------------------
# 1. Load P-1 telemetry
# --------------------------------------------------

matrix, channel_data, failures = load_all_channels()

channel = "P-1"

train = channel_data[channel]["train_telemetry"]
test = channel_data[channel]["test_telemetry"]


# --------------------------------------------------
# 2. Load ground-truth anomaly intervals
# --------------------------------------------------

annotations = pd.read_csv("labeled_anomalies.csv")

row = annotations[
    (annotations["chan_id"] == channel) &
    (annotations["spacecraft"] == "SMAP")
]

if len(row) != 1:
    raise ValueError(
        f"Expected exactly one P-1 annotation row, found {len(row)}"
    )

sequences = ast.literal_eval(
    row.iloc[0]["anomaly_sequences"]
)

y_true = np.zeros(len(test), dtype=int)

for start, end in sequences:
    if start < 0 or end >= len(test) or start > end:
        raise ValueError(
            f"Invalid anomaly interval: {start}, {end}"
        )

    # Annotation endpoints are inclusive.
    y_true[start:end + 1] = 1


# --------------------------------------------------
# 3. Train temporal detector
# --------------------------------------------------

split = int(len(train) * 0.8)

fit_data = train[:split]
calibration_data = train[split:]

detector = TemporalAnomalyDetector(window=10)

detector.fit(fit_data)

calibration_scores = detector.score(
    calibration_data,
    history=fit_data[-10:]
)

threshold = detector.calibrate_threshold(
    calibration_scores,
    percentile=99
)


# --------------------------------------------------
# 4. Score and predict test telemetry
# --------------------------------------------------

test_scores = detector.score(
    test,
    history=train[-10:]
)

y_pred = detector.predict(
    test,
    threshold=threshold,
    history=train[-10:]
)


# --------------------------------------------------
# 5. Calculate evaluation metrics
# --------------------------------------------------

tn, fp, fn, tp = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
).ravel()

precision = precision_score(
    y_true, y_pred, zero_division=0
)

recall = recall_score(
    y_true, y_pred, zero_division=0
)

f1 = f1_score(
    y_true, y_pred, zero_division=0
)

roc_auc = roc_auc_score(y_true, test_scores)

avg_precision = average_precision_score(
    y_true, test_scores
)


# --------------------------------------------------
# 6. Print results
# --------------------------------------------------

print("=" * 50)
print("TEMPORAL DETECTOR EVALUATION:", channel)
print("=" * 50)

print("\nGround-truth samples:", np.sum(y_true))
print("Predicted anomaly samples:", np.sum(y_pred))

print("\nConfusion Matrix:")
print("TP:", tp)
print("FP:", fp)
print("TN:", tn)
print("FN:", fn)

print("\nEvaluation Metrics:")
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1-score:", round(f1, 4))
print("ROC-AUC:", round(roc_auc, 4))
print("Average Precision:", round(avg_precision, 4))

print("\nDetection Coverage:")
print(
    "Anomalous samples detected:",
    f"{tp}/{np.sum(y_true)}"
)

print(
    "Normal samples incorrectly flagged:",
    fp
)

print("\nThreshold:", round(threshold, 6))
print("=" * 50)
