
import ast
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, average_precision_score

from ashm_fdir.data_loader import load_all_channels
from ashm_fdir.detectors.temporal_features import (
    extract_temporal_features
)
from ashm_fdir.detectors.temporal_detector import (
    TemporalAnomalyDetector
)

# Load P-1 data.
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

# Fit detector using training data only.
split = int(len(train) * 0.8)

fit_data = train[:split]
calibration_data = train[split:]

detector = TemporalAnomalyDetector(window=10)
detector.fit(fit_data)

# Extract test features.
features = extract_temporal_features(
    test,
    window=10,
    history=train[-10:]
)

# Calculate each feature's standardized deviation.
component_scores = np.abs(
    features - detector.feature_medians_
) / detector.feature_scales_

names = [
    "Rolling Mean",
    "Rolling Standard Deviation",
    "First Difference"
]

print("\nFEATURE-WISE ANOMALY ANALYSIS")
print("=" * 65)

for i, name in enumerate(names):

    normal_scores = component_scores[y_true == 0, i]
    anomaly_scores = component_scores[y_true == 1, i]

    auc = roc_auc_score(
        y_true,
        component_scores[:, i]
    )

    ap = average_precision_score(
        y_true,
        component_scores[:, i]
    )

    print(f"\n{name}")
    print("-" * 40)

    print("Normal median:", np.median(normal_scores))
    print("Anomaly median:", np.median(anomaly_scores))

    print(
        "Normal 95th percentile:",
        np.percentile(normal_scores, 95)
    )

    print(
        "Anomaly 95th percentile:",
        np.percentile(anomaly_scores, 95)
    )

    print("ROC-AUC:", round(auc, 4))
    print("Average Precision:", round(ap, 4))

# Inspect which feature dominates the combined score.
dominant_feature = np.argmax(component_scores, axis=1)

print("\nDOMINANT FEATURE CONTRIBUTION")
print("=" * 65)

for i, name in enumerate(names):
    print(
        name,
        ":",
        np.sum(dominant_feature == i),
        "samples"
    )
