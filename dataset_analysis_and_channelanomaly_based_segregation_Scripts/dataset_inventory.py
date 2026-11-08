# ==========================================
# SECTION 1: IMPORTS AND CONFIGURATION
# ==========================================

import os
import ast
import numpy as np
import pandas as pd
from pathlib import Path


# Project directory
BASE_DIR = Path(__file__).resolve().parent

# Dataset directories
TRAIN_DIR = BASE_DIR / "data" / "train"
TEST_DIR = BASE_DIR / "data" / "test"

# Anomaly annotation file
ANNOTATION_FILE = BASE_DIR / "labeled_anomalies.csv"


# Check whether required paths exist
for path in [TRAIN_DIR, TEST_DIR, ANNOTATION_FILE]:
    if not path.exists():
        raise FileNotFoundError(f"Required path not found: {path}")

print("Configuration loaded successfully.")
print("Training directory:", TRAIN_DIR)
print("Testing directory:", TEST_DIR)
print("Annotation file:", ANNOTATION_FILE)

# ==========================================
# SECTION 2: DISCOVER TELEMETRY FILES
# ==========================================

train_files = {
    file.stem: file
    for file in TRAIN_DIR.glob("*.npy")
}

test_files = {
    file.stem: file
    for file in TEST_DIR.glob("*.npy")
}

print("\n========== DATASET FILE DISCOVERY ==========")

print("Training channels found:", len(train_files))
print("Testing channels found:", len(test_files))

print("\nFirst 10 training channels:")
print(list(train_files.keys())[:10])

print("\nFirst 10 testing channels:")
print(list(test_files.keys())[:10])


# Compare channel availability
train_channels = set(train_files.keys())
test_channels = set(test_files.keys())

print("\n========== CHANNEL AVAILABILITY ==========")

print("Present in both:", len(train_channels & test_channels))
print("Missing from training:", sorted(test_channels - train_channels))
print("Missing from testing:", sorted(train_channels - test_channels))

# ==========================================
# SECTION 3: LOAD ANOMALY ANNOTATIONS
# ==========================================

annotations = pd.read_csv(ANNOTATION_FILE)

print("\n========== ANNOTATION CSV ==========")

print("Total annotation rows:", len(annotations))

print("\nColumns:")
print(annotations.columns.tolist())

print("\nFirst 5 annotation records:")
print(annotations.head().to_string(index=False))


# Check for duplicate channel entries
duplicate_channels = annotations[
    annotations.duplicated(subset=["chan_id"], keep=False)
].sort_values("chan_id")

print("\n========== DUPLICATE ANNOTATIONS ==========")

if duplicate_channels.empty:
    print("No duplicate channel entries found.")
else:
    print(duplicate_channels.to_string(index=False))


# Count annotation categories
print("\n========== ANOMALY CLASS DISTRIBUTION ==========")

print(annotations["class"].value_counts(dropna=False))

print("\n========== SPACECRAFT DISTRIBUTION ==========")

print(annotations["spacecraft"].value_counts(dropna=False))

# ==========================================
# SECTION 4: PARSE AND VALIDATE ANNOTATIONS
# ==========================================

# Convert CSV string representations into Python lists
annotations["parsed_sequences"] = annotations["anomaly_sequences"].apply(
    ast.literal_eval
)

# Parse class labels from the CSV
annotations["parsed_classes"] = annotations["class"].apply(
    lambda value: [
        label.strip()
        for label in value.strip("[]").split(",")
    ]
)

print("\n========== ANNOTATION VALIDATION ==========")

# Check every annotation row against its test telemetry file
for _, row in annotations.iterrows():

    channel = row["chan_id"]
    sequences = row["parsed_sequences"]

    if channel not in test_files:
        print(f"{channel}: TEST FILE MISSING")
        continue

    # Load original test telemetry
    test_data = np.load(test_files[channel])

    actual_length = len(test_data)
    declared_length = int(row["num_values"])

    # Check declared length against actual telemetry length
    length_status = (
        "OK" if actual_length == declared_length
        else "LENGTH MISMATCH"
    )

    # Check anomaly interval boundaries
    invalid_sequences = []

    for start, end in sequences:
        if start < 0 or end < start or end >= actual_length:
            invalid_sequences.append([start, end])

    print(
        f"{channel}: "
        f"Shape={test_data.shape}, "
        f"Declared={declared_length}, "
        f"Length={length_status}, "
        f"Invalid intervals={invalid_sequences}"
    )

# ==========================================
# SECTION 5: DATASET-WIDE SUMMARY
# ==========================================

# Remove duplicate annotation records temporarily
# We will investigate P-2 separately.
unique_annotations = annotations.drop_duplicates(
    subset=["chan_id"],
    keep="first"
).copy()


# Count anomaly types at channel level
def get_channel_type(classes):

    has_point = "point" in classes
    has_contextual = "contextual" in classes

    if has_point and has_contextual:
        return "Mixed"

    elif has_point:
        return "Point"

    elif has_contextual:
        return "Contextual"

    return "Unknown"


unique_annotations["anomaly_type"] = (
    unique_annotations["parsed_classes"].apply(get_channel_type)
)


print("\n========== DATASET-WIDE SUMMARY ==========")

print("Total telemetry channels:", len(train_files))
print("Unique annotated channels:", len(unique_annotations))

print("\n========== SPACECRAFT SUMMARY ==========")

print(unique_annotations["spacecraft"].value_counts())

print("\n========== ANOMALY TYPE SUMMARY ==========")

print(unique_annotations["anomaly_type"].value_counts())


# Count individual anomaly events and labelled samples
total_events = 0
total_anomalous_samples = 0

for _, row in unique_annotations.iterrows():

    sequences = row["parsed_sequences"]

    total_events += len(sequences)

    for start, end in sequences:
        total_anomalous_samples += end - start + 1


print("\n========== ANOMALY STATISTICS ==========")

print("Total anomaly events:", total_events)

print(
    "Total labelled anomalous samples:",
    total_anomalous_samples
)


# Count events by spacecraft
print("\n========== EVENTS BY SPACECRAFT ==========")

for spacecraft in unique_annotations["spacecraft"].unique():

    subset = unique_annotations[
        unique_annotations["spacecraft"] == spacecraft
    ]

    event_count = sum(
        len(sequences)
        for sequences in subset["parsed_sequences"]
    )

    print(f"{spacecraft}: {event_count} events")

# ==========================================
# SECTION 6: PER-CHANNEL INVENTORY
# ==========================================

inventory_rows = []

for channel in sorted(set(train_files) | set(test_files)):

    # Get annotation records for this channel
    channel_annotations = annotations[
        annotations["chan_id"] == channel
    ]

    # Default values for unannotated channels
    spacecraft = "Unknown"
    anomaly_type = "Unannotated"
    event_count = 0
    labelled_samples = 0
    unique_anomalous_samples = 0
    annotation_status = "Missing annotation"

    if not channel_annotations.empty:

        # Use the first record for provisional statistics
        row = channel_annotations.iloc[0]

        spacecraft = row["spacecraft"]
        anomaly_type = get_channel_type(row["parsed_classes"])
        annotation_status = "OK"

        sequences = row["parsed_sequences"]

        event_count = len(sequences)

        labelled_samples = sum(
            end - start + 1
            for start, end in sequences
        )

        # Merge overlapping or adjacent intervals
        merged = []

        for start, end in sorted(sequences):

            if not merged or start > merged[-1][1] + 1:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)

        unique_anomalous_samples = sum(
            end - start + 1
            for start, end in merged
        )

        # Flag duplicate annotation records
        if len(channel_annotations) > 1:
            annotation_status = "Duplicate annotation"

    # Read telemetry shapes
    train_length = (
        len(np.load(train_files[channel], mmap_mode="r"))
        if channel in train_files else None
    )

    test_data = (
        np.load(test_files[channel], mmap_mode="r")
        if channel in test_files else None
    )

    test_length = len(test_data) if test_data is not None else None
    feature_count = test_data.shape[1] if test_data is not None else None

    inventory_rows.append({
        "channel": channel,
        "spacecraft": spacecraft,
        "anomaly_type": anomaly_type,
        "train_length": train_length,
        "test_length": test_length,
        "features": feature_count,
        "events": event_count,
        "labelled_samples": labelled_samples,
        "unique_anomalous_samples": unique_anomalous_samples,
        "annotation_status": annotation_status
    })


# Convert results into a DataFrame
inventory_df = pd.DataFrame(inventory_rows)

print("\n========== PER-CHANNEL INVENTORY ==========")

print(inventory_df.to_string(index=False))


# Save inventory to CSV
output_file = BASE_DIR / "dataset_inventory.csv"

inventory_df.to_csv(output_file, index=False)

print("\nInventory saved to:", output_file)

# ==========================================
# SECTION 7: TELEMETRY STATISTICAL CHARACTERIZATION
# ==========================================

def calculate_signal_statistics(data):

    # Extract primary telemetry measurement
    signal = np.asarray(data[:, 0], dtype=np.float64)

    # Count invalid values
    nan_count = int(np.isnan(signal).sum())
    inf_count = int(np.isinf(signal).sum())

    # Exclude invalid values from statistical calculations
    valid_signal = signal[np.isfinite(signal)]

    if len(valid_signal) == 0:
        return {
            "samples": len(signal),
            "valid_samples": 0,
            "nan_count": nan_count,
            "inf_count": inf_count,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "unique_values": 0
        }

    return {
        "samples": len(signal),
        "valid_samples": len(valid_signal),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "mean": float(np.mean(valid_signal)),
        "std": float(np.std(valid_signal)),
        "min": float(np.min(valid_signal)),
        "max": float(np.max(valid_signal)),
        "unique_values": int(np.unique(valid_signal).size)
    }


# Store statistics for every channel
statistics_rows = []

for channel in sorted(set(train_files) | set(test_files)):

    train_data = np.load(
        train_files[channel], mmap_mode="r"
    )

    test_data = np.load(
        test_files[channel], mmap_mode="r"
    )

    train_stats = calculate_signal_statistics(train_data)
    test_stats = calculate_signal_statistics(test_data)

    statistics_rows.append({

        "channel": channel,

        "train_mean": train_stats["mean"],
        "train_std": train_stats["std"],
        "train_min": train_stats["min"],
        "train_max": train_stats["max"],
        "train_unique": train_stats["unique_values"],
        "train_nan": train_stats["nan_count"],
        "train_inf": train_stats["inf_count"],

        "test_mean": test_stats["mean"],
        "test_std": test_stats["std"],
        "test_min": test_stats["min"],
        "test_max": test_stats["max"],
        "test_unique": test_stats["unique_values"],
        "test_nan": test_stats["nan_count"],
        "test_inf": test_stats["inf_count"],

        "train_constant": train_stats["unique_values"] <= 1,
        "test_constant": test_stats["unique_values"] <= 1

    })


statistics_df = pd.DataFrame(statistics_rows)

print("\n========== TELEMETRY STATISTICS ==========")

print(statistics_df.to_string(index=False))


# Save statistics
statistics_file = BASE_DIR / "telemetry_statistics.csv"

statistics_df.to_csv(statistics_file, index=False)

print("\nStatistics saved to:", statistics_file)

# ==========================================
# SECTION 7B: STATISTICAL SUMMARY
# ==========================================

print("\n========== SIGNAL CHARACTERISTICS ==========")

print(
    "Constant training channels:",
    statistics_df["train_constant"].sum()
)

print(
    "Constant testing channels:",
    statistics_df["test_constant"].sum()
)

print(
    "Channels with NaN values:",
    (
        (statistics_df["train_nan"] > 0) |
        (statistics_df["test_nan"] > 0)
    ).sum()
)

print(
    "Channels with infinite values:",
    (
        (statistics_df["train_inf"] > 0) |
        (statistics_df["test_inf"] > 0)
    ).sum()
)


print("\n========== CHANNELS WITH CONSTANT TRAINING SIGNAL ==========")

print(
    statistics_df[
        statistics_df["train_constant"]
    ][["channel", "train_unique", "test_unique", "train_std", "test_std"]]
    .to_string(index=False)
)


print("\n========== TRAIN/TEST DISTRIBUTION COMPARISON ==========")

statistics_df["mean_difference"] = (
    statistics_df["test_mean"] - statistics_df["train_mean"]
)

statistics_df["std_difference"] = (
    statistics_df["test_std"] - statistics_df["train_std"]
)

print(
    statistics_df[
        ["channel", "mean_difference", "std_difference"]
    ].to_string(index=False)
)

# ==========================================
# SECTION 8: CONSTANT CHANNEL INVESTIGATION
# ==========================================

constant_channels = statistics_df[
    statistics_df["train_constant"]
].copy()

print("\n========== CONSTANT TRAINING CHANNEL ANALYSIS ==========")

for _, row in constant_channels.iterrows():

    channel = row["channel"]

    train_data = np.load(train_files[channel])
    test_data = np.load(test_files[channel])

    train_signal = train_data[:, 0]
    test_signal = test_data[:, 0]

    print(f"\nChannel: {channel}")

    print("Training value:", train_signal[0])
    print("Training samples:", len(train_signal))

    print("Testing unique values:", np.unique(test_signal).size)
    print("Testing mean:", np.mean(test_signal))
    print("Testing standard deviation:", np.std(test_signal))
    print("Testing minimum:", np.min(test_signal))
    print("Testing maximum:", np.max(test_signal))