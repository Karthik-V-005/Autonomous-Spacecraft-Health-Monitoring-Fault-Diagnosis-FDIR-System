
import numpy as np

from ashm_fdir.data_loader import load_all_channels
from ashm_fdir.detectors.temporal_features import (
    extract_temporal_features
)

# Load the telemetry dataset.
matrix, channel_data, failures = load_all_channels()

# Start with P-1, where the statistical detector struggled.
channel = "P-1"

data = channel_data[channel]

train = data["train_telemetry"]
test = data["test_telemetry"]

# Extract features from training telemetry.
train_features = extract_temporal_features(
    train,
    window=10
)

# Extract test features using the end of training as historical context.
test_features = extract_temporal_features(
    test,
    window=10,
    history=train[-10:]
)

print("Channel:", channel)
print("Training telemetry shape:", train.shape)
print("Training feature shape:", train_features.shape)
print("Test telemetry shape:", test.shape)
print("Test feature shape:", test_features.shape)

feature_names = [
    "Rolling Mean",
    "Rolling Standard Deviation",
    "First Difference"
]

print("\nTraining feature statistics:")

for i, name in enumerate(feature_names):
    print(
        f"{name}: "
        f"Mean={np.mean(train_features[:, i]):.6f}, "
        f"Std={np.std(train_features[:, i]):.6f}"
    )

print("\nFirst five test feature rows:")
print(test_features[:5])
