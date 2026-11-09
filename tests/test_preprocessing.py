
import numpy as np

from ashm_fdir.data_loader import load_channel
from ashm_fdir.preprocessing import (
    ChannelPreprocessor,
    preprocessing_summary
)


# Test representative channels:
# A-1: Constant training telemetry
# P-1: Contextual anomalies
# F-3: Strong auxiliary feature differences
# T-9: MSL channel with short training history

CHANNELS = ["A-1", "P-1", "F-3", "T-9"]


print("\n========================================")
print("AURA PREPROCESSING VALIDATION")
print("========================================")

for channel in CHANNELS:

    print(f"\nProcessing {channel}")

    data = load_channel(channel)

    train_data = data["train_data"]
    test_data = data["test_data"]

    preprocessor = ChannelPreprocessor()

    # Fit using training data only.
    train_scaled = preprocessor.fit_transform(train_data)

    # Transform test data using the same fitted scaler.
    test_scaled = preprocessor.transform(test_data)

    summary = preprocessing_summary(preprocessor)

    print("Original train shape:", train_data.shape)
    print("Scaled train shape:  ", train_scaled.shape)

    print("Original test shape: ", test_data.shape)
    print("Scaled test shape:   ", test_scaled.shape)

    print("Telemetry constant:", summary["telemetry_constant"])

    print(
        "Constant auxiliary features:",
        summary["auxiliary_constant_features"]
    )

    print(
        "Training scaled median:",
        round(float(np.median(train_scaled[:, 0])), 5)
    )

    print(
        "Training scaled IQR:",
        round(
            float(
                np.percentile(train_scaled[:, 0], 75)
                - np.percentile(train_scaled[:, 0], 25)
            ),
            5
        )
    )

    assert train_scaled.shape == train_data.shape
    assert test_scaled.shape == test_data.shape

    assert np.isfinite(train_scaled).all()
    assert np.isfinite(test_scaled).all()

    print("Validation: PASSED")

print("\n========================================")
print("ALL PREPROCESSING TESTS COMPLETED")
print("========================================")
