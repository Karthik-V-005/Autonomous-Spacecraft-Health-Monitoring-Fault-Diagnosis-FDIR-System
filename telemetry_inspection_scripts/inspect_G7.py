import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Load G-7 data
# --------------------------------------------------

TRAIN_PATH = "../data/train/G-7.npy"
TEST_PATH = "../data/test/G-7.npy"

train_data = np.load(TRAIN_PATH)
test_data = np.load(TEST_PATH)

train_telemetry = train_data[:, 0]
test_telemetry = test_data[:, 0]

print("Training shape:", train_data.shape)
print("Testing shape:", test_data.shape)

# --------------------------------------------------
# 2. Define anomaly intervals
# --------------------------------------------------

anomalies = [
    (3650, 3750, "Contextual A1"),
    (5050, 5100, "Point A2"),
    (7560, 7675, "Contextual A3")
]

# --------------------------------------------------
# 3. Calculate training statistics
# --------------------------------------------------

print("\nTraining Telemetry Statistics")
print("--------------------------------")

print("Mean:", np.mean(train_telemetry))
print("Standard deviation:", np.std(train_telemetry))
print("Minimum:", np.min(train_telemetry))
print("Maximum:", np.max(train_telemetry))
print("Unique values:", len(np.unique(train_telemetry)))

# --------------------------------------------------
# 4. Compare anomaly and nearby normal regions
# --------------------------------------------------

def print_statistics(name, values):

    print(f"\n{name}")
    print("-" * 40)

    print("Count:", len(values))
    print("Mean:", np.mean(values))
    print("Standard deviation:", np.std(values))
    print("Minimum:", np.min(values))
    print("Maximum:", np.max(values))
    print("Median:", np.median(values))


for start, end, name in anomalies:

    anomaly_values = test_telemetry[start:end + 1]

    # Use nearby normal regions, avoiding other anomalies
    normal_start = max(0, start - 200)
    normal_end = min(len(test_telemetry), end + 201)

    normal_mask = np.ones(
        normal_end - normal_start,
        dtype=bool
    )

    for other_start, other_end, _ in anomalies:
        overlap_start = max(normal_start, other_start)
        overlap_end = min(normal_end, other_end)

        if overlap_start <= overlap_end:
            normal_mask[
                overlap_start - normal_start:
                overlap_end - normal_start + 1
            ] = False

    nearby_normal = test_telemetry[
        normal_start:normal_end
    ][normal_mask]

    print_statistics(name, anomaly_values)
    print_statistics(name + " Nearby Normal", nearby_normal)

# --------------------------------------------------
# 5. Plot complete G-7 telemetry
# --------------------------------------------------

plt.figure(figsize=(16, 5))

plt.plot(
    test_telemetry,
    linewidth=0.7,
    label="G-7 telemetry"
)

for start, end, name in anomalies:

    plt.axvspan(
        start,
        end,
        alpha=0.25,
        label=name
    )

plt.title("G-7 Complete Test Telemetry")
plt.xlabel("Timestep")
plt.ylabel("Telemetry value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 6. Plot each anomaly separately
# --------------------------------------------------

for start, end, name in anomalies:

    zoom_start = max(0, start - 100)
    zoom_end = min(len(test_telemetry), end + 101)

    plt.figure(figsize=(14, 4))

    plt.plot(
        range(zoom_start, zoom_end),
        test_telemetry[zoom_start:zoom_end],
        label="Telemetry",
        linewidth=0.9
    )

    plt.axvspan(
        start,
        end,
        color="orange",
        alpha=0.3,
        label="Ground-truth anomaly"
    )

    plt.title(f"G-7: {name}")
    plt.xlabel("Timestep")
    plt.ylabel("Telemetry value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --------------------------------------------------
# 7. Rolling statistics
# --------------------------------------------------

series = pd.Series(test_telemetry)

rolling_mean = series.rolling(window=10).mean()
rolling_std = series.rolling(window=10).std()

for start, end, name in anomalies:

    zoom_start = max(0, start - 100)
    zoom_end = min(len(test_telemetry), end + 101)

    fig, axes = plt.subplots(
        2, 1,
        figsize=(14, 7)
    )

    axes[0].plot(
        range(zoom_start, zoom_end),
        rolling_mean.iloc[zoom_start:zoom_end],
        label="Rolling mean"
    )

    axes[0].axvspan(
        start, end,
        color="orange",
        alpha=0.3
    )

    axes[0].set_title(f"{name} - Rolling Mean")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(
        range(zoom_start, zoom_end),
        rolling_std.iloc[zoom_start:zoom_end],
        label="Rolling standard deviation"
    )

    axes[1].axvspan(
        start, end,
        color="orange",
        alpha=0.3
    )

    axes[1].set_title(f"{name} - Rolling Standard Deviation")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()