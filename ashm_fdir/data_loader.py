
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# SECTION 1: PROJECT PATHS
# ============================================================

# This file is inside aura_engine.
# Its parent is aura_engine; parent.parent is the project root.

PROJECT_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = PROJECT_DIR / "data" / "train"
TEST_DIR = PROJECT_DIR / "data" / "test"

MATRIX_FILE = (
    PROJECT_DIR
    / "characterization_results"
    / "detector_selection_matrix.csv"
)


# ============================================================
# SECTION 2: LOAD CHANNEL CONFIGURATION
# ============================================================

def load_channel_matrix():

    if not MATRIX_FILE.exists():
        raise FileNotFoundError(
            f"Detector matrix not found: {MATRIX_FILE}"
        )

    matrix = pd.read_csv(MATRIX_FILE)

    required_columns = [
        "channel",
        "spacecraft",
        "annotation_status",
        "anomaly_classes",
        "primary_detector",
        "secondary_detector",
        "train_constant"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in matrix.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing matrix columns: {missing_columns}"
        )

    if matrix["channel"].duplicated().any():
        duplicates = matrix.loc[
            matrix["channel"].duplicated(),
            "channel"
        ].tolist()

        raise ValueError(
            f"Duplicate channels in detector matrix: {duplicates}"
        )

    return matrix


# ============================================================
# SECTION 3: LOAD INDIVIDUAL TELEMETRY ARRAYS
# ============================================================

def load_channel(channel):

    train_path = TRAIN_DIR / f"{channel}.npy"
    test_path = TEST_DIR / f"{channel}.npy"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training file not found: {train_path}"
        )

    if not test_path.exists():
        raise FileNotFoundError(
            f"Testing file not found: {test_path}"
        )

    train_data = np.load(
        train_path,
        allow_pickle=False
    )

    test_data = np.load(
        test_path,
        allow_pickle=False
    )

    # Both arrays must be two-dimensional.
    if train_data.ndim != 2 or test_data.ndim != 2:
        raise ValueError(
            f"{channel}: Expected 2D arrays. "
            f"Train shape: {train_data.shape}, "
            f"Test shape: {test_data.shape}"
        )

    # Feature count must agree between train and test.
    if train_data.shape[1] != test_data.shape[1]:
        raise ValueError(
            f"{channel}: Train/test feature mismatch. "
            f"Train: {train_data.shape[1]}, "
            f"Test: {test_data.shape[1]}"
        )

    if train_data.shape[1] < 1:
        raise ValueError(
            f"{channel}: No telemetry column found."
        )

    # Column 0 is the primary telemetry signal.
    # Remaining columns are auxiliary features.

    train_telemetry = train_data[:, 0]
    test_telemetry = test_data[:, 0]

    train_auxiliary = train_data[:, 1:]
    test_auxiliary = test_data[:, 1:]

    return {
        "channel": channel,

        "train_data": train_data,
        "test_data": test_data,

        "train_telemetry": train_telemetry,
        "test_telemetry": test_telemetry,

        "train_auxiliary": train_auxiliary,
        "test_auxiliary": test_auxiliary,

        "train_shape": train_data.shape,
        "test_shape": test_data.shape,

        "train_finite_count": int(
            np.isfinite(train_data).sum()
        ),

        "test_finite_count": int(
            np.isfinite(test_data).sum()
        ),

        "train_nonfinite_count": int(
            (~np.isfinite(train_data)).sum()
        ),

        "test_nonfinite_count": int(
            (~np.isfinite(test_data)).sum()
        )
    }


# ============================================================
# SECTION 4: LOAD ALL CHANNELS
# ============================================================

def load_all_channels():

    matrix = load_channel_matrix()

    channel_data = {}
    failures = []

    for channel in matrix["channel"]:

        try:
            channel_data[channel] = load_channel(channel)

        except Exception as error:
            failures.append({
                "channel": channel,
                "error": str(error)
            })

    return matrix, channel_data, failures


# ============================================================
# SECTION 5: DATASET VALIDATION REPORT
# ============================================================

def print_validation_report(matrix, channel_data, failures):

    print("\n========================================")
    print("AURA DATA LOADER VALIDATION")
    print("========================================")

    print(f"Channels in detector matrix: {len(matrix)}")
    print(f"Successfully loaded: {len(channel_data)}")
    print(f"Failed to load: {len(failures)}")

    total_train = sum(
        item["train_shape"][0]
        for item in channel_data.values()
    )

    total_test = sum(
        item["test_shape"][0]
        for item in channel_data.values()
    )

    print(f"\nTotal training timesteps: {total_train}")
    print(f"Total testing timesteps: {total_test}")

    total_train_nonfinite = sum(
        item["train_nonfinite_count"]
        for item in channel_data.values()
    )

    total_test_nonfinite = sum(
        item["test_nonfinite_count"]
        for item in channel_data.values()
    )

    print(
        f"\nNon-finite training values: "
        f"{total_train_nonfinite}"
    )

    print(
        f"Non-finite testing values: "
        f"{total_test_nonfinite}"
    )

    print("\nChannel loading details:")

    for channel, item in channel_data.items():

        print(
            f"{channel:6} "
            f"Train: {str(item['train_shape']):14} "
            f"Test: {str(item['test_shape']):14} "
            f"Non-finite: "
            f"{item['train_nonfinite_count'] + item['test_nonfinite_count']}"
        )

    if failures:

        print("\nFailed channels:")

        for failure in failures:
            print(
                f"{failure['channel']}: "
                f"{failure['error']}"
            )

    print("\n========================================")
    print("VALIDATION COMPLETE")
    print("========================================")


# ============================================================
# SECTION 6: DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    matrix, channel_data, failures = load_all_channels()

    print_validation_report(
        matrix,
        channel_data,
        failures
    )
