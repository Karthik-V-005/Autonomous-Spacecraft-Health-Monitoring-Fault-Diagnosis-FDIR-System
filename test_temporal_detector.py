
import numpy as np

from ashm_fdir.data_loader import load_all_channels
from ashm_fdir.detectors.temporal_detector import (
    TemporalAnomalyDetector
)

# Load P-1 telemetry.
matrix, channel_data, failures = load_all_channels()

channel = "P-1"

train = channel_data[channel]["train_telemetry"]
test = channel_data[channel]["test_telemetry"]

# Chronological 80/20 training split.
split = int(len(train) * 0.8)

fit_data = train[:split]
calibration_data = train[split:]

# Fit the detector on the first 80%.
detector = TemporalAnomalyDetector(window=10)
detector.fit(fit_data)

# Score the remaining 20%, retaining preceding context.
calibration_scores = detector.score(
    calibration_data,
    history=fit_data[-10:]
)

# Calibrate the threshold on training data only.
threshold = detector.calibrate_threshold(
    calibration_scores,
    percentile=99
)

# Score test telemetry, using the end of training as context.
test_scores = detector.score(
    test,
    history=train[-10:]
)

predictions = detector.predict(
    test,
    threshold=threshold,
    history=train[-10:]
)

print("Channel:", channel)
print("Fit samples:", len(fit_data))
print("Calibration samples:", len(calibration_data))
print("Test samples:", len(test))

print("\nThreshold:", threshold)
print("Test score shape:", test_scores.shape)
print("Prediction shape:", predictions.shape)

print("\nTest score statistics:")
print("Minimum:", np.min(test_scores))
print("Median:", np.median(test_scores))
print("95th percentile:", np.percentile(test_scores, 95))
print("Maximum:", np.max(test_scores))

print("\nPredicted anomalies:", np.sum(predictions))
print(
    "Predicted anomaly percentage:",
    round(np.mean(predictions) * 100, 3),
    "%"
)
