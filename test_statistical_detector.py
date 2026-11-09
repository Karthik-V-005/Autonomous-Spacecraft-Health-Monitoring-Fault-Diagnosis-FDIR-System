
import numpy as np

from ashm_fdir.detectors.statistical_detector import (
    RobustStatisticalDetector
)


# Simulated normal telemetry
train_data = np.random.normal(
    loc=10,
    scale=1,
    size=1000
)

# Validation data for threshold calibration
validation_data = np.random.normal(
    loc=10,
    scale=1,
    size=200
)

# Test data with an artificial anomaly
test_data = np.array([
    9.8,
    10.2,
    10.5,
    9.5,
    18.0,
    10.1,
    2.0
])


detector = RobustStatisticalDetector()

detector.fit(train_data)

validation_scores = detector.score(validation_data)

threshold = detector.calibrate_threshold(
    validation_scores,
    percentile=99
)

predictions = detector.predict(test_data)

scores = detector.score(test_data)


print("Detector:", detector.__class__.__name__)
print("Training median:", detector.median_)
print("Training MAD:", detector.mad_)
print("Calibration threshold:", threshold)

print("\nTest results:")

for value, score, prediction in zip(
    test_data,
    scores,
    predictions
):
    print(
        f"Value: {value:6.2f} | "
        f"Score: {score:8.3f} | "
        f"Prediction: {prediction}"
    )
