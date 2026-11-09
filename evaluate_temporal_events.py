
import ast
import numpy as np
import pandas as pd

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

# Read labelled anomaly intervals.
annotations = pd.read_csv("labeled_anomalies.csv")

row = annotations[
    (annotations["chan_id"] == "P-1") &
    (annotations["spacecraft"] == "SMAP")
]

sequences = ast.literal_eval(
    row.iloc[0]["anomaly_sequences"]
)

# Train using the first 80% of training data.
split = int(len(train) * 0.8)

fit_data = train[:split]
calibration_data = train[split:]

detector = TemporalAnomalyDetector(window=10)
detector.fit(fit_data)

# Extract temporal features.
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

# Use rolling standard deviation only.
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

# Evaluate the predeclared calibration percentiles.
percentiles = [95, 97, 98, 99]

for percentile in percentiles:

    threshold = np.percentile(
        calibration_scores,
        percentile
    )

    predictions = (test_scores > threshold).astype(int)

    print("\n" + "=" * 60)
    print(f"EVENT EVALUATION - {percentile}th PERCENTILE")
    print("Threshold:", round(threshold, 4))
    print("=" * 60)

    detected_events = 0

    for event_number, (start, end) in enumerate(sequences, 1):

        event_predictions = predictions[start:end + 1]

        detected_samples = int(np.sum(event_predictions))
        event_length = end - start + 1

        coverage = detected_samples / event_length

        detected = detected_samples > 0

        if detected:
            detected_events += 1

        print(f"\nEvent {event_number}: {start} - {end}")
        print("Event length:", event_length)
        print("Detected samples:", detected_samples)
        print("Coverage:", round(coverage * 100, 2), "%")
        print("Event detected:", detected)

    print(
        "\nEvent detection rate:",
        f"{detected_events}/{len(sequences)}"
    )
