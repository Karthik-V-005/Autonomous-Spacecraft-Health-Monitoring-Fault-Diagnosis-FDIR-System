import numpy as np

# Load datasets
train_data = np.load("../data/train/G-7.npy")
test_data = np.load("../data/test/G-7.npy")

train = train_data[:, 0]
test = test_data[:, 0]

# Anomaly intervals
anomalies = [
    (3650, 3750, "A1 Contextual"),
    (5050, 5100, "A2 Point"),
    (7560, 7675, "A3 Contextual")
]


def show_distribution(name, values):

    unique, counts = np.unique(
        values,
        return_counts=True
    )

    print(f"\n{name}")
    print("-" * 45)

    for value, count in zip(unique, counts):

        print(
            f"Value: {value:8.3f} | "
            f"Count: {count:5d} | "
            f"Percentage: {count / len(values) * 100:.2f}%"
        )


# Training distribution
show_distribution(
    "TRAINING TELEMETRY",
    train
)

# Complete test distribution
show_distribution(
    "COMPLETE TEST TELEMETRY",
    test
)

# Individual anomaly distributions
for start, end, name in anomalies:

    show_distribution(
        name,
        test[start:end + 1]
    )

# Nearby normal distributions
for start, end, name in anomalies:

    normal_start = max(0, start - 200)
    normal_end = min(len(test), end + 201)

    nearby = np.concatenate([
        test[normal_start:start],
        test[end + 1:normal_end]
    ])

    show_distribution(
        name + " - NEARBY NORMAL",
        nearby
    )