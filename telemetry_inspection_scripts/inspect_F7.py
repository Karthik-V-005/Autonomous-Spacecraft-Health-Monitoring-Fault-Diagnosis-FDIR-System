
import numpy as np

# --------------------------------------------------
# 1. Load F-7 data
# --------------------------------------------------

TRAIN_PATH = "../data/train/F-7.npy"
TEST_PATH = "../data/test/F-7.npy"

train_data = np.load(TRAIN_PATH).astype(np.float32)
test_data = np.load(TEST_PATH).astype(np.float32)

train = train_data[:, 0]
test = test_data[:, 0]

print("F-7 DATASET INSPECTION")
print("=" * 55)

print("Training shape:", train_data.shape)
print("Testing shape:", test_data.shape)

print("\nTraining telemetry shape:", train.shape)
print("Testing telemetry shape:", test.shape)


# --------------------------------------------------
# 2. Basic statistical information
# --------------------------------------------------

print("\nTRAINING TELEMETRY STATISTICS")
print("=" * 55)

print("Mean:", np.mean(train))
print("Standard deviation:", np.std(train))
print("Minimum:", np.min(train))
print("Maximum:", np.max(train))
print("Unique values:", len(np.unique(train)))

print("\nTEST TELEMETRY STATISTICS")
print("=" * 55)

print("Mean:", np.mean(test))
print("Standard deviation:", np.std(test))
print("Minimum:", np.min(test))
print("Maximum:", np.max(test))
print("Unique values:", len(np.unique(test)))


# --------------------------------------------------
# 3. Value distribution
# --------------------------------------------------

def show_distribution(data, name):

    values, counts = np.unique(
        data,
        return_counts=True
    )

    print(f"\n{name} VALUE DISTRIBUTION")
    print("-" * 55)

    for value, count in zip(values, counts):

        percentage = count / len(data) * 100

        print(
            f"Value: {value:.6f} | "
            f"Count: {count} | "
            f"Percentage: {percentage:.2f}%"
        )


show_distribution(train, "TRAINING")
show_distribution(test, "TESTING")


# --------------------------------------------------
# 4. Ground-truth anomaly regions
# --------------------------------------------------

ANOMALIES = [
    (1250, 1450, "A1 Contextual"),
    (2670, 2790, "A2 Contextual"),
    (3325, 3425, "A3 Contextual")
]

print("\nANOMALY REGION STATISTICS")
print("=" * 55)

for start, end, name in ANOMALIES:

    anomaly = test[start:end + 1]

    print(f"\n{name}: {start} - {end}")

    print("Length:", len(anomaly))
    print("Mean:", np.mean(anomaly))
    print("Standard deviation:", np.std(anomaly))
    print("Minimum:", np.min(anomaly))
    print("Maximum:", np.max(anomaly))

    show_distribution(anomaly, name)


# --------------------------------------------------
# 5. Compare anomaly regions with nearby normal data
# --------------------------------------------------

print("\nANOMALY VS NEARBY NORMAL")
print("=" * 55)

for start, end, name in ANOMALIES:

    normal_start = max(0, start - 100)
    normal_end = min(len(test), end + 101)

    normal_before = test[normal_start:start]
    normal_after = test[end + 1:normal_end]

    nearby_normal = np.concatenate([
        normal_before,
        normal_after
    ])

    anomaly = test[start:end + 1]

    print(f"\n{name}")

    print(
        "Nearby normal mean:",
        np.mean(nearby_normal)
    )

    print(
        "Anomaly mean:",
        np.mean(anomaly)
    )

    print(
        "Nearby normal std:",
        np.std(nearby_normal)
    )

    print(
        "Anomaly std:",
        np.std(anomaly)
    )
