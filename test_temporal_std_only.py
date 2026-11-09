
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
from ashm_fdir.detectors.temporal_features import (
    extract_temporal_features
)
from ashm_fdir.detectors.temporal_detector import (
    TemporalAnomalyDetector
)

# 1. Load P-1 telemetry.
matrix, channel_data, failures = load_all_channels()

train = channel_data["P-1"]["train_telemetry"]
test = channel_data["P-1"]["test_telemetry"]

# 2. Load ground-truth annotations.
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

# 3. Split training data chronologically.
split = int(len(train) * 0.8)

fit_data = train[:split]
calibration_data = train[split:]

# 4. Fit the existing temporal feature statistics.
detector = TemporalAnomalyDetector(window=10)
detector.fit(fit_data)

# 5. Extract calibration features with historical context.
calibration_features = extract_temporal_features(
    calibration_data,
    window=10,
    history=fit_data[-10:]
)

# 6. Extract test features with historical context.
test_features = extract_temporal_features(
    test,
    window=10,
    history=train[-10:]
)

# 7. Calculate rolling-standard-deviation deviations only.
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

# 8. Calibrate threshold using training calibration data.
threshold = np.percentile(calibration_scores, 99)

# 9. Generate predictions.
y_pred = (test_scores > threshold).astype(int)

# 10. Evaluate.
tn, fp, fn, tp = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
).ravel()

print("=" * 55)
print("ROLLING STANDARD DEVIATION ONLY - P-1")
print("=" * 55)

print("\nThreshold:", threshold)
print("Ground-truth anomalies:", np.sum(y_true))
print("Predicted anomalies:", np.sum(y_pred))

print("\nConfusion Matrix:")
print("TP:", tp)
print("FP:", fp)
print("TN:", tn)
print("FN:", fn)

print("\nMetrics:")
print("Precision:", round(
    precision_score(y_true, y_pred, zero_division=0), 4
))
print("Recall:", round(
    recall_score(y_true, y_pred, zero_division=0), 4
))
print("F1-score:", round(
    f1_score(y_true, y_pred, zero_division=0), 4
))
print("ROC-AUC:", round(
    roc_auc_score(y_true, test_scores), 4
))
print("Average Precision:", round(
    average_precision_score(y_true, test_scores), 4
))

print("\nScore statistics:")
print("Normal median:", np.median(test_scores[y_true == 0]))
print("Anomaly median:", np.median(test_scores[y_true == 1]))

print("=" * 55)
