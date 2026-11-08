# ==========================================
# SECTION 1: CONFIGURATION
# ==========================================

import ast
import numpy as np
import pandas as pd

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

TRAIN_DIR = BASE_DIR / "data" / "train"
TEST_DIR = BASE_DIR / "data" / "test"

ANNOTATION_FILE = BASE_DIR / "labeled_anomalies.csv"

OUTPUT_DIR = BASE_DIR / "characterization_results"
OUTPUT_DIR.mkdir(exist_ok=True)


annotations = pd.read_csv(ANNOTATION_FILE)


def parse_sequences(value):
    return ast.literal_eval(value)


def parse_classes(value):
    return [
        label.strip()
        for label in value.strip("[]").split(",")
        if label.strip()
    ]


annotations["parsed_sequences"] = (
    annotations["anomaly_sequences"].apply(parse_sequences)
)

annotations["parsed_classes"] = (
    annotations["class"].apply(parse_classes)
)


train_files = {
    file.stem: file
    for file in TRAIN_DIR.glob("*.npy")
}

test_files = {
    file.stem: file
    for file in TEST_DIR.glob("*.npy")
}


print("Dataset characterization initialized.")
print("Training channels:", len(train_files))
print("Testing channels:", len(test_files))
print("Annotation records:", len(annotations))

# ==========================================
# SECTION 2: ANOMALY EVENT CHARACTERIZATION
# ==========================================

event_rows = []

for _, row in annotations.iterrows():

    channel = row["chan_id"]

    if channel not in test_files:
        continue

    test_data = np.load(
        test_files[channel],
        mmap_mode="r"
    )

    signal = np.asarray(test_data[:, 0], dtype=np.float64)

    sequences = row["parsed_sequences"]
    classes = row["parsed_classes"]

    for event_index, (start, end) in enumerate(sequences):

        # Extract event telemetry
        event_signal = signal[start:end + 1]

        # Identify corresponding class
        if event_index < len(classes):
            event_class = classes[event_index]
        else:
            event_class = "Unknown"

        event_rows.append({

            "channel": channel,
            "spacecraft": row["spacecraft"],
            "event_number": event_index + 1,
            "event_class": event_class,

            "start": start,
            "end": end,

            "duration": len(event_signal),

            "mean": np.mean(event_signal),
            "std": np.std(event_signal),

            "minimum": np.min(event_signal),
            "maximum": np.max(event_signal),

            "unique_values": np.unique(event_signal).size

        })


event_df = pd.DataFrame(event_rows)

event_file = OUTPUT_DIR / "anomaly_event_characterization.csv"

event_df.to_csv(event_file, index=False)


print("\n========== ANOMALY EVENT CHARACTERIZATION ==========")

print("Total event records:", len(event_df))

print("\nEvent class distribution:")
print(event_df["event_class"].value_counts())

print("\nEvent duration statistics:")
print(event_df["duration"].describe())

print("\nLongest 10 anomaly events:")

print(
    event_df.nlargest(10, "duration")[
        ["channel", "event_number", "event_class", "duration"]
    ].to_string(index=False)
)

print("\nSaved:", event_file)

# ==========================================
# SECTION 3: TEMPORAL BEHAVIOUR ANALYSIS
# ==========================================

def analyze_temporal_behavior(signal):

    signal = np.asarray(signal, dtype=np.float64)

    # Identify changes between consecutive samples
    changes = np.diff(signal) != 0

    transition_count = int(np.sum(changes))

    # Identify lengths of consecutive identical-value runs
    change_points = np.where(changes)[0] + 1

    boundaries = np.concatenate((
        [0],
        change_points,
        [len(signal)]
    ))

    run_lengths = np.diff(boundaries)

    return {

        "samples": len(signal),

        "transitions": transition_count,

        "transition_rate": (
            transition_count / (len(signal) - 1)
            if len(signal) > 1 else 0
        ),

        "unique_values": np.unique(signal).size,

        "run_count": len(run_lengths),

        "mean_run_length": np.mean(run_lengths),

        "maximum_run_length": np.max(run_lengths),

        "minimum_run_length": np.min(run_lengths)

    }


temporal_rows = []

all_channels = sorted(set(train_files) | set(test_files))

for channel in all_channels:

    train_data = np.load(
        train_files[channel],
        mmap_mode="r"
    )

    test_data = np.load(
        test_files[channel],
        mmap_mode="r"
    )

    train_signal = train_data[:, 0]
    test_signal = test_data[:, 0]

    train_stats = analyze_temporal_behavior(train_signal)
    test_stats = analyze_temporal_behavior(test_signal)

    temporal_rows.append({

        "channel": channel,

        "train_samples": train_stats["samples"],
        "train_transitions": train_stats["transitions"],
        "train_transition_rate": train_stats["transition_rate"],
        "train_unique_values": train_stats["unique_values"],
        "train_mean_run": train_stats["mean_run_length"],
        "train_max_run": train_stats["maximum_run_length"],

        "test_samples": test_stats["samples"],
        "test_transitions": test_stats["transitions"],
        "test_transition_rate": test_stats["transition_rate"],
        "test_unique_values": test_stats["unique_values"],
        "test_mean_run": test_stats["mean_run_length"],
        "test_max_run": test_stats["maximum_run_length"]

    })


temporal_df = pd.DataFrame(temporal_rows)

temporal_file = OUTPUT_DIR / "temporal_behavior_analysis.csv"

temporal_df.to_csv(temporal_file, index=False)


print("\n========== TEMPORAL BEHAVIOUR SUMMARY ==========")

print(
    temporal_df[
        [
            "test_transitions",
            "test_transition_rate",
            "test_mean_run",
            "test_max_run"
        ]
    ].describe()
)

print("\nHighest transition counts:")

print(
    temporal_df.nlargest(10, "test_transitions")[
        ["channel", "test_transitions", "test_transition_rate"]
    ].to_string(index=False)
)

print("\nLongest stable runs:")

print(
    temporal_df.nlargest(10, "test_max_run")[
        ["channel", "test_max_run", "test_unique_values"]
    ].to_string(index=False)
)

print("\nSaved:", temporal_file)