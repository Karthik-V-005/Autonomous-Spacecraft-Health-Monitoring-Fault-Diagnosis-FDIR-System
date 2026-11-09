
import ast
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

from ashm_fdir.data_loader import load_all_channels
from ashm_fdir.detectors.temporal_features import (
    extract_temporal_features
)
from ashm_fdir.detectors.temporal_detector import (
    TemporalAnomalyDetector
)

# Load P-1 telemetry.
matrix, channel_data, failures = load_all_channels()

train = channel_data["P-1"]["train_telemetry"]
test = channel_data["P-1"]["test_telemetry"]

# Load ground-truth labels.
annotations = pd.read_csv("labeled_anomalies.csv")

row = annotations[
    (annotations["chan_id"] == "P-1") &
    (annotations["spacecraft"] == "SMAP")
]

sequences = ast.literal_eval(
    row.iloc[0]["anomaly_sequences"]
)

y_true = np.zeros(len(test), dtype=int)

for start, end in sequences:
    y_true[start:end + 1] = 1

# Chronological training split.
split = int(len(train) * 0.8)

fit_data = train[:split]
calibration_data = train[split:]

# Fit detector.
detector = TemporalAnomalyDetector(window=10)
detector.fit(fit_data)

# Extract rolling standard deviation features.
calibration_features = extract_temporal_features(
    calibration_data,
    window=10,
    history=fit_data[-10:]
)

test_features = extract_temporal_features(
    test,
    window=10,
    history=train[-10:]
)

# Calculate standardized rolling standard deviation scores.
calibration_scores = (
    np.abs(
        calibration_features[:, 1] -
        detector.feature_medians_[1]
    ) / detector.feature_scales_[1]
)

test_scores = (
    np.abs(
        test_features[:, 1] -
        detector.feature_medians_[1]
    ) / detector.feature_scales_[1]
)

# Thresholds determined exclusively from calibration scores.
percentiles = [95, 97, 98, 99]

print("\nTEMPORAL DETECTOR - THRESHOLD COMPARISON")
print("=" * 85)

print(
    f"{'Percentile':<12}"
    f"{'Threshold':<13}"
    f"{'Predicted':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
)

for percentile in percentiles:

    threshold = np.percentile(
        calibration_scores,
        percentile
    )

    y_pred = (test_scores > threshold).astype(int)

    precision = precision_score(
        y_true, y_pred, zero_division=0
    )

    recall = recall_score(
        y_true, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_true, y_pred, zero_division=0
    )

    print(
        f"{percentile:<12}"
        f"{threshold:<13.4f}"
        f"{np.sum(y_pred):<12}"
        f"{precision:<12.4f}"
        f"{recall:<12.4f}"
        f"{f1:<12.4f}"
    )

print("\nThreshold-independent metrics:")
print("ROC-AUC:", round(roc_auc_score(y_true, test_scores), 4))
print("Average Precision:", round(
    average_precision_score(y_true, test_scores), 4
))

print("\nCalibration sample count:", len(calibration_scores))
print("=" * 85)
