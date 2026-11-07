import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --------------------------------------------------
# 1. Load A-1 test data
# --------------------------------------------------

TEST_PATH = "../data/test/A-1.npy"

data = np.load(TEST_PATH)

print("Dataset shape:", data.shape)

# Column 0 contains the telemetry measurement
telemetry = data[:, 0]

# Ground-truth anomaly interval (inclusive)
ANOMALY_START = 4690
ANOMALY_END = 4774

# --------------------------------------------------
# 2. Extract anomaly and normal regions
# --------------------------------------------------

anomaly = telemetry[ANOMALY_START:ANOMALY_END + 1]

# Nearby normal regions before and after the anomaly
normal_before = telemetry[4500:ANOMALY_START]
normal_after = telemetry[ANOMALY_END + 1:4950]

nearby_normal = np.concatenate([normal_before, normal_after])

# --------------------------------------------------
# 3. Print statistical comparison
# --------------------------------------------------

def print_statistics(name, values):
    print(f"\n{name}")
    print("-" * 35)
    print("Count:", len(values))
    print("Mean:", np.mean(values))
    print("Standard deviation:", np.std(values))
    print("Minimum:", np.min(values))
    print("Maximum:", np.max(values))
    print("Median:", np.median(values))

print_statistics("A-1 Anomaly", anomaly)
print_statistics("Nearby Normal Telemetry", nearby_normal)

# --------------------------------------------------
# 3B. Inspect actual unusual values
# --------------------------------------------------

unusual_indices = np.where(
    telemetry[ANOMALY_START:ANOMALY_END + 1] != 1.0
)[0] + ANOMALY_START

print("\nUnusual values inside A-1 anomaly:")
print("Number of unusual samples:", len(unusual_indices))

for index in unusual_indices:
    print(
        f"Timestep: {index}, "
        f"Value: {telemetry[index]}"
    )

print("\nUnique values inside anomaly:")
print(np.unique(anomaly, return_counts=True))

# --------------------------------------------------
# 4. Calculate rolling statistics
# --------------------------------------------------

series = pd.Series(telemetry)

rolling_mean = series.rolling(window=10).mean()
rolling_std = series.rolling(window=10).std()

# --------------------------------------------------
# 5. Plot the complete telemetry
# --------------------------------------------------

plt.figure(figsize=(15, 5))

plt.plot(telemetry, linewidth=0.7, label="A-1 Telemetry")

plt.axvspan(
    ANOMALY_START,
    ANOMALY_END,
    color="orange",
    alpha=0.3,
    label="Ground-truth anomaly"
)

plt.title("A-1 Complete Test Telemetry")
plt.xlabel("Timestep")
plt.ylabel("Telemetry value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 6. Zoom into the anomaly
# --------------------------------------------------

ZOOM_START = 4500
ZOOM_END = 4950

plt.figure(figsize=(15, 5))

plt.plot(
    range(ZOOM_START, ZOOM_END),
    telemetry[ZOOM_START:ZOOM_END],
    label="Telemetry",
    linewidth=1
)

plt.axvspan(
    ANOMALY_START,
    ANOMALY_END,
    color="orange",
    alpha=0.3,
    label="Ground-truth anomaly"
)

plt.title("A-1 Telemetry Around the Anomaly")
plt.xlabel("Timestep")
plt.ylabel("Telemetry value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 7. Plot rolling mean and standard deviation
# --------------------------------------------------

fig, axes = plt.subplots(2, 1, figsize=(15, 8))

axes[0].plot(
    range(ZOOM_START, ZOOM_END),
    rolling_mean.iloc[ZOOM_START:ZOOM_END],
    label="Rolling mean (10 timesteps)"
)
axes[0].axvspan(ANOMALY_START, ANOMALY_END, color="orange", alpha=0.3)
axes[0].set_title("A-1 Rolling Mean")
axes[0].set_xlabel("Timestep")
axes[0].set_ylabel("Rolling mean")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(
    range(ZOOM_START, ZOOM_END),
    rolling_std.iloc[ZOOM_START:ZOOM_END],
    label="Rolling standard deviation (10 timesteps)"
)
axes[1].axvspan(ANOMALY_START, ANOMALY_END, color="orange", alpha=0.3)
axes[1].set_title("A-1 Rolling Standard Deviation")
axes[1].set_xlabel("Timestep")
axes[1].set_ylabel("Rolling standard deviation")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

# --------------------------------------------------
# 8. Inspect training telemetry distribution
# --------------------------------------------------

train_data = np.load("../data/train/A-1.npy")
train_telemetry = train_data[:, 0]

print("\nA-1 Training Telemetry Distribution")
print("------------------------------------")

values, counts = np.unique(
    train_telemetry,
    return_counts=True
)

for value, count in zip(values, counts):
    print(f"Value: {value}, Count: {count}")

print("\nTraining samples:", len(train_telemetry))
print("Unique telemetry values:", len(values))