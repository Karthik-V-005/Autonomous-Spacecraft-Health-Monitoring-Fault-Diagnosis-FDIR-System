# ==========================================
# SECTION 9: VISUAL INSPECTION
# ==========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = BASE_DIR / "data" / "train"
TEST_DIR = BASE_DIR / "data" / "test"

ANNOTATION_FILE = BASE_DIR / "labeled_anomalies.csv"


# Channels selected for investigation
CHANNELS = ["A-1", "C-2", "M-6", "D-2", "P-4"]


# Load annotations
annotations = pd.read_csv(ANNOTATION_FILE)


def parse_sequences(value):

    import ast
    return ast.literal_eval(value)


# Plot each channel
for channel in CHANNELS:

    train_data = np.load(TRAIN_DIR / f"{channel}.npy")
    test_data = np.load(TEST_DIR / f"{channel}.npy")

    train_signal = train_data[:, 0]
    test_signal = test_data[:, 0]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Training plot
    axes[0].plot(train_signal, linewidth=0.8)

    axes[0].set_title(
        f"{channel} - Training Telemetry"
    )

    axes[0].set_ylabel("Telemetry value")
    axes[0].grid(alpha=0.3)

    # Testing plot
    axes[1].plot(test_signal, linewidth=0.8)

    axes[1].set_title(
        f"{channel} - Testing Telemetry"
    )

    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel("Telemetry value")
    axes[1].grid(alpha=0.3)

    # Add ground-truth anomaly intervals
    channel_rows = annotations[
        annotations["chan_id"] == channel
    ]

    if not channel_rows.empty:

        sequences = parse_sequences(
            channel_rows.iloc[0]["anomaly_sequences"]
        )

        for start, end in sequences:

            axes[1].axvspan(
                start,
                end,
                alpha=0.2,
                color="red"
            )

    plt.tight_layout()
    plt.show()