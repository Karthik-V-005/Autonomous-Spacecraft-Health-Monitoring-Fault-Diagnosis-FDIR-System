
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# SECTION 1: CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

TEST_DIR = BASE_DIR / "data" / "test"
ANNOTATION_FILE = BASE_DIR / "labeled_anomalies.csv"

OUTPUT_DIR = BASE_DIR / "characterization_results"
OUTPUT_DIR.mkdir(exist_ok=True)

CHANNELS = {
    "F-3": [1, 2, 21, 22],
    "T-9": [5, 13, 14],
    "D-14": [13, 14],
    "E-13": [5, 6]
}

# ============================================================
# SECTION 2: LOAD ANNOTATIONS
# ============================================================

annotations = pd.read_csv(ANNOTATION_FILE)

def parse_sequences(value):
    return ast.literal_eval(value)

def get_channel_intervals(channel):
    rows = annotations[annotations["chan_id"] == channel]

    if rows.empty:
        raise ValueError(f"No annotation found for {channel}")

    # These four channels have no duplicate annotation rows.
    sequences = parse_sequences(rows.iloc[0]["anomaly_sequences"])

    return sequences

# ============================================================
# SECTION 3: LOAD DATA AND PREPARE PLOTS
# ============================================================

fig, axes = plt.subplots(
    nrows=4,
    ncols=2,
    figsize=(18, 16)
)

fig.suptitle(
    "AURA: Comparative Telemetry and Auxiliary Feature Analysis",
    fontsize=17,
    fontweight="bold"
)

for row, (channel, feature_numbers) in enumerate(CHANNELS.items()):

    print(f"\nProcessing {channel}...")

    file_path = TEST_DIR / f"{channel}.npy"

    if not file_path.exists():
        print(f"File not found: {file_path}")
        continue

    data = np.load(file_path)

    telemetry = data[:, 0]
    time = np.arange(len(telemetry))

    intervals = get_channel_intervals(channel)

    # --------------------------------------------------------
    # SECTION 4: PRIMARY TELEMETRY PLOT
    # --------------------------------------------------------

    ax1 = axes[row, 0]

    ax1.plot(
        time,
        telemetry,
        linewidth=0.8,
        label="Primary telemetry"
    )

    for start, end in intervals:
        ax1.axvspan(
            start,
            end,
            alpha=0.2,
            color="red"
        )

    ax1.set_title(f"{channel} - Primary Telemetry")
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Telemetry value")
    ax1.grid(alpha=0.3)

    # --------------------------------------------------------
    # SECTION 5: AUXILIARY FEATURE PLOT
    # --------------------------------------------------------

    ax2 = axes[row, 1]

    for feature_number in feature_numbers:

        # feature_1 corresponds to data[:, 1]
        column_index = feature_number

        if column_index >= data.shape[1]:
            print(
                f"Skipping feature_{feature_number}: "
                f"column unavailable"
            )
            continue

        feature_values = data[:, column_index]

        ax2.plot(
            time,
            feature_values,
            linewidth=0.9,
            label=f"Feature {feature_number}"
        )

    for start, end in intervals:
        ax2.axvspan(
            start,
            end,
            alpha=0.2,
            color="red"
        )

    ax2.set_title(
        f"{channel} - Selected Auxiliary Features"
    )

    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Feature value")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper right", fontsize=8)

    print(f"  Data shape: {data.shape}")
    print(f"  Labelled intervals: {intervals}")
    print(f"  Features plotted: {feature_numbers}")

# ============================================================
# SECTION 6: FINALIZE AND SAVE
# ============================================================

plt.tight_layout(rect=[0, 0, 1, 0.96])

output_path = OUTPUT_DIR / "four_channel_comparison.png"

plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print("\n========================================")
print("FOUR-CHANNEL COMPARISON COMPLETE")
print("========================================")
print(f"Saved visualization to: {output_path}")
