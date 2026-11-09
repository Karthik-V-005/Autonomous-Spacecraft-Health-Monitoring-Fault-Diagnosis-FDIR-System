
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# Load anomaly intervals.
annotations = pd.read_csv("labeled_anomalies.csv")

row = annotations[
    (annotations["chan_id"] == "P-1") &
    (annotations["spacecraft"] == "SMAP")
]

sequences = ast.literal_eval(
    row.iloc[0]["anomaly_sequences"]
)

# Train detector using chronological split.
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

# Extract test features and component deviations.
features = extract_temporal_features(
    test,
    window=10,
    history=train[-10:]
)

component_scores = np.abs(
    features - detector.feature_medians_
) / detector.feature_scales_

total_scores = np.max(component_scores, axis=1)

# Create ground-truth labels.
labels = np.zeros(len(test), dtype=int)

for start, end in sequences:
    labels[start:end + 1] = 1

# Plot the telemetry and each feature score.
fig, axes = plt.subplots(
    5, 1,
    figsize=(16, 15),
    sharex=True
)

axes[0].plot(test, linewidth=0.7)
axes[0].set_title("P-1 Test Telemetry")

feature_names = [
    "Rolling Mean Deviation",
    "Rolling Standard Deviation Deviation",
    "First Difference Deviation"
]

for i in range(3):
    axes[i + 1].plot(
        component_scores[:, i],
        linewidth=0.7
    )
    axes[i + 1].set_title(feature_names[i])

axes[4].plot(
    total_scores,
    linewidth=0.7,
    label="Combined anomaly score"
)

axes[4].axhline(
    threshold,
    linestyle="--",
    label="Calibration threshold"
)

axes[4].set_title("Combined Temporal Anomaly Score")
axes[4].legend()

# Shade labelled anomaly intervals on every plot.
for ax in axes:
    for start, end in sequences:
        ax.axvspan(
            start, end,
            color="red",
            alpha=0.12
        )

    ax.grid(alpha=0.2)

axes[-1].set_xlabel("Test timestep")

plt.tight_layout()

plt.savefig(
    "temporal_detector_models_P1.png",
    dpi=150
)

plt.show()

print("Threshold:", threshold)
print("Feature scales:", detector.feature_scales_)
print("Feature medians:", detector.feature_medians_)
print("Saved temporal detector diagnostic plot.")
