# ==========================================
# SECTION 1: CONFIGURATION AND DATA LOADING
# ==========================================

import ast
import numpy as np
import pandas as pd

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

TRAIN_DIR = BASE_DIR / "data" / "train"
TEST_DIR = BASE_DIR / "data" / "test"

ANNOTATION_FILE = BASE_DIR / "labeled_anomalies.csv"

OUTPUT_DIR = BASE_DIR / "multivariate_results"
OUTPUT_DIR.mkdir(exist_ok=True)


annotations = pd.read_csv(ANNOTATION_FILE)

train_files = {
    file.stem: file
    for file in TRAIN_DIR.glob("*.npy")
}

test_files = {
    file.stem: file
    for file in TEST_DIR.glob("*.npy")
}


def parse_sequences(value):
    return ast.literal_eval(value)


annotations["parsed_sequences"] = (
    annotations["anomaly_sequences"].apply(parse_sequences)
)


print("Multivariate analysis initialized.")
print("Training channels:", len(train_files))
print("Testing channels:", len(test_files))
print("Annotation records:", len(annotations))

# ==========================================
# SECTION 2: ANOMALY MASK GENERATION
# ==========================================

def create_anomaly_mask(data_length, sequences):

    mask = np.zeros(data_length, dtype=bool)

    for start, end in sequences:

        if start < 0 or end >= data_length or end < start:
            continue

        # Inclusive anomaly intervals
        mask[start:end + 1] = True

    return mask

# ==========================================
# SECTION 3: FEATURE RELATIONSHIP ANALYSIS
# ==========================================

feature_rows = []
channel_rows = []

for channel in sorted(set(train_files) | set(test_files)):

    train_data = np.load(
        train_files[channel],
        mmap_mode="r"
    )

    test_data = np.load(
        test_files[channel],
        mmap_mode="r"
    )

    train_data = np.asarray(train_data, dtype=np.float64)
    test_data = np.asarray(test_data, dtype=np.float64)

    telemetry = test_data[:, 0]
    auxiliary = test_data[:, 1:]

    channel_annotations = annotations[
        annotations["chan_id"] == channel
    ]

    # Generate anomaly mask if annotation exists
    if not channel_annotations.empty:

        annotation = channel_annotations.iloc[0]

        anomaly_mask = create_anomaly_mask(
            len(test_data),
            annotation["parsed_sequences"]
        )

        annotation_status = (
            "Duplicate - provisional first record"
            if len(channel_annotations) > 1
            else "Annotated"
        )

    else:

        anomaly_mask = np.zeros(len(test_data), dtype=bool)
        annotation_status = "Unannotated"

    normal_mask = ~anomaly_mask

    # Compare only finite values
    for feature_index in range(1, test_data.shape[1]):

        feature = test_data[:, feature_index]

        valid = (
            np.isfinite(feature) &
            np.isfinite(telemetry)
        )

        normal_valid = valid & normal_mask
        anomaly_valid = valid & anomaly_mask

        normal_values = feature[normal_valid]
        anomaly_values = feature[anomaly_valid]

        # Skip invalid or empty comparisons
        if len(normal_values) == 0 or len(anomaly_values) == 0:
            continue

        normal_mean = np.mean(normal_values)
        anomaly_mean = np.mean(anomaly_values)

        normal_std = np.std(normal_values)
        anomaly_std = np.std(anomaly_values)

        # Pooled standard deviation
        pooled_std = np.sqrt(
            (normal_std ** 2 + anomaly_std ** 2) / 2
        )

        if pooled_std > 1e-12:
            effect_size = (
                anomaly_mean - normal_mean
            ) / pooled_std
        else:
            effect_size = 0.0

        # Correlation with primary telemetry
        correlation = np.nan

        feature_valid = feature[valid]
        telemetry_valid = telemetry[valid]

        if (
            len(feature_valid) > 1 and
            np.std(feature_valid) > 1e-12 and
            np.std(telemetry_valid) > 1e-12
        ):

            correlation = np.corrcoef(
                feature_valid,
                telemetry_valid
            )[0, 1]

        feature_rows.append({

            "channel": channel,

            "feature": f"feature_{feature_index}",

            "anomaly_status": annotation_status,

            "normal_samples": len(normal_values),

            "anomaly_samples": len(anomaly_values),

            "normal_mean": normal_mean,

            "anomaly_mean": anomaly_mean,

            "normal_std": normal_std,

            "anomaly_std": anomaly_std,

            "mean_difference": anomaly_mean - normal_mean,

            "standardized_difference": effect_size,

            "telemetry_correlation": correlation

        })

    channel_rows.append({

        "channel": channel,

        "total_features": auxiliary.shape[1],

        "anomaly_samples": int(anomaly_mask.sum()),

        "normal_samples": int(normal_mask.sum()),

        "annotation_status": annotation_status

    })


feature_df = pd.DataFrame(feature_rows)
channel_df = pd.DataFrame(channel_rows)


feature_file = (
    OUTPUT_DIR / "feature_relationship_analysis.csv"
)

channel_file = (
    OUTPUT_DIR / "multivariate_channel_summary.csv"
)


feature_df.to_csv(feature_file, index=False)
channel_df.to_csv(channel_file, index=False)


print("\nFeature relationship analysis completed.")

print("Feature records:", len(feature_df))

print("Saved:", feature_file)
print("Saved:", channel_file)

# ==========================================
# SECTION 4: MULTIVARIATE SUMMARY
# ==========================================

print("\n========== MULTIVARIATE ANALYSIS SUMMARY ==========")

print(
    "Channels analyzed:",
    len(channel_df)
)

print(
    "Total auxiliary feature comparisons:",
    len(feature_df)
)

print(
    "Annotated channels:",
    (channel_df["annotation_status"] == "Annotated").sum()
)

print(
    "Unannotated channels:",
    (channel_df["annotation_status"] == "Unannotated").sum()
)


# Strongest feature differences
print("\n========== STRONGEST FEATURE DIFFERENCES ==========")

feature_df["absolute_effect"] = (
    feature_df["standardized_difference"].abs()
)

print(
    feature_df.nlargest(20, "absolute_effect")[
        [
            "channel",
            "feature",
            "standardized_difference",
            "telemetry_correlation",
            "anomaly_status"
        ]
    ].to_string(index=False)
)


# Strongest telemetry correlations
print("\n========== STRONGEST TELEMETRY CORRELATIONS ==========")

correlation_df = feature_df.dropna(
    subset=["telemetry_correlation"]
).copy()

correlation_df["absolute_correlation"] = (
    correlation_df["telemetry_correlation"].abs()
)

print(
    correlation_df.nlargest(20, "absolute_correlation")[
        [
            "channel",
            "feature",
            "telemetry_correlation"
        ]
    ].to_string(index=False)
)


# Channels with most informative-looking features
print("\n========== CHANNEL FEATURE SUMMARY ==========")

channel_effects = (
    feature_df.groupby("channel")["absolute_effect"]
    .max()
    .reset_index(name="maximum_absolute_effect")
    .sort_values("maximum_absolute_effect", ascending=False)
)

print(channel_effects.head(20).to_string(index=False))