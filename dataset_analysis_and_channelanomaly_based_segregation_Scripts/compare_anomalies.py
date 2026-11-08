import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

TEST_DIR = "data/test"
LABEL_FILE = "labeled_anomalies.csv"


# ============================================================
# LOAD LABELS
# ============================================================

labels = pd.read_csv(LABEL_FILE)

# Convert anomaly_sequences from string → Python list
labels["anomaly_sequences"] = labels["anomaly_sequences"].apply(
    ast.literal_eval
)

# Clean class column
def parse_classes(value):
    value = value.strip("[]")
    return [x.strip() for x in value.split(",") if x.strip()]

labels["class"] = labels["class"].apply(parse_classes)


# ============================================================
# CHANNELS WE WANT TO STUDY
# ============================================================

channels = [
    "A-1",
    "P-1",
    "G-7",
    "P-4",
    "F-7"
]


# ============================================================
# PLOT EACH CHANNEL
# ============================================================

for channel in channels:

    # Find label information
    row = labels[labels["chan_id"] == channel]

    if row.empty:
        print(f"No label found for {channel}")
        continue

    row = row.iloc[0]

    spacecraft = row["spacecraft"]
    sequences = row["anomaly_sequences"]
    classes = row["class"]

    # Load test data
    file_path = f"{TEST_DIR}/{channel}.npy"

    try:
        data = np.load(file_path)
    except FileNotFoundError:
        print(f"Test file not found: {file_path}")
        continue

    telemetry = data[:, 0]

    print("\n===================================")
    print("Channel:", channel)
    print("Spacecraft:", spacecraft)
    print("Shape:", data.shape)
    print("Anomaly classes:", classes)
    print("Anomaly sequences:", sequences)

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(
        telemetry,
        linewidth=1,
        label=f"{channel} telemetry"
    )

    # Add anomaly regions
    for sequence, anomaly_class in zip(sequences, classes):

        start, end = sequence

        if anomaly_class == "point":
            label = "Point anomaly"
        else:
            label = "Contextual anomaly"

        ax.axvspan(
            start,
            end,
            alpha=0.25,
            label=label
        )

    ax.set_title(
        f"{channel} — {spacecraft} Telemetry"
    )

    ax.set_xlabel("Time step")
    ax.set_ylabel("Telemetry value")

    ax.grid(True)

    # Remove duplicate legend entries
    handles, legend_labels = ax.get_legend_handles_labels()

    unique = dict(zip(legend_labels, handles))

    ax.legend(unique.values(), unique.keys())

    plt.tight_layout()
    plt.show()