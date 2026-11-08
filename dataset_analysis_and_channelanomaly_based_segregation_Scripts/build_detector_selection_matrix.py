
import ast
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# SECTION 1: PATHS AND INPUT FILES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "characterization_results"
OUTPUT_DIR.mkdir(exist_ok=True)


def locate_file(filename):
    """Find a generated CSV in the project root or output folder."""

    possible_paths = [
        BASE_DIR / filename,
        OUTPUT_DIR / filename
    ]

    for path in possible_paths:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Could not find {filename} in project root or "
        f"{OUTPUT_DIR}"
    )


statistics_file = locate_file("telemetry_statistics.csv")
temporal_file = locate_file("temporal_behavior_analysis.csv")
annotations_file = locate_file("labeled_anomalies.csv")
features_file = BASE_DIR / "multivariate_results" / "feature_relationship_analysis.csv"

if not features_file.exists():
    raise FileNotFoundError(
        f"Feature analysis file not found: {features_file}"
    )


# ============================================================
# SECTION 2: LOAD EXISTING ANALYSIS RESULTS
# ============================================================

statistics = pd.read_csv(statistics_file)
temporal = pd.read_csv(temporal_file)
annotations = pd.read_csv(annotations_file)
features = pd.read_csv(features_file)

print("Loaded existing analysis files.")
print(f"Channels in statistics: {len(statistics)}")
print(f"Channels in temporal analysis: {len(temporal)}")


# ============================================================
# SECTION 3: PREPARE ANNOTATION INFORMATION
# ============================================================

def parse_sequences(value):
    return ast.literal_eval(value)


def parse_classes(value):
    """
    Convert strings such as:
    [point]
    [contextual, point]
    into Python lists.
    """

    cleaned = str(value).strip()

    cleaned = cleaned.strip("[]")

    if not cleaned:
        return []

    return [
        item.strip().lower()
        for item in cleaned.split(",")
    ]


annotation_info = {}

duplicate_channels = set(
    annotations.loc[
        annotations["chan_id"].duplicated(keep=False),
        "chan_id"
    ]
)

# Keep the first annotation record provisionally.
# P-2 has conflicting duplicate records and requires review.

annotations_unique = annotations.drop_duplicates(
    subset="chan_id",
    keep="first"
)

for _, row in annotations_unique.iterrows():

    channel = row["chan_id"]

    sequences = parse_sequences(
        row["anomaly_sequences"]
    )

    classes = parse_classes(row["class"])

    annotation_info[channel] = {
        "spacecraft": row["spacecraft"],
        "sequences": sequences,
        "classes": classes,
        "event_count": len(sequences)
    }


# ============================================================
# SECTION 4: PREPARE AUXILIARY FEATURE EVIDENCE
# ============================================================

feature_summary = {}

for channel, group in features.groupby("channel"):

    valid = group.dropna(
        subset=["standardized_difference",
                "telemetry_correlation"]
    )

    if valid.empty:
        feature_summary[channel] = {
            "strong_feature_count": 0,
            "max_abs_effect": np.nan,
            "max_abs_correlation": np.nan,
            "candidate_features": ""
        }
        continue

    strong = valid[
        valid["standardized_difference"].abs() >= 0.2
    ]

    top_features = (
        valid.assign(
            abs_effect=valid["standardized_difference"].abs()
        )
        .sort_values("abs_effect", ascending=False)
        .head(5)["feature"]
        .tolist()
    )

    feature_summary[channel] = {
        "strong_feature_count": len(strong),
        "max_abs_effect": valid[
            "standardized_difference"
        ].abs().max(),
        "max_abs_correlation": valid[
            "telemetry_correlation"
        ].abs().max(),
        "candidate_features": ", ".join(top_features)
    }


# ============================================================
# SECTION 5: DETECTOR-SELECTION RULES
# ============================================================

def choose_detector(
    train_constant,
    transition_rate,
    anomaly_classes
):

    classes = set(anomaly_classes)

    if not classes:
        return (
            "Unsupervised baseline monitoring",
            "Statistical monitoring / change detection",
            "No labelled anomaly intervals available"
        )

    if classes == {"point", "contextual"}:
        return (
            "Hybrid detector",
            "Statistical + temporal detection",
            "Contains both point and contextual labels"
        )

    if classes == {"contextual"}:

        if train_constant:
            return (
                "Contextual state-change detector",
                "Run-length + window-based detection",
                "Contextual labels with constant training baseline"
            )

        return (
            "Temporal/contextual detector",
            "Window statistics or forecasting",
            "Contextual anomaly labels"
        )

    if classes == {"point"}:

        if train_constant:
            return (
                "Baseline-deviation detector",
                "State-change detection",
                "Point labels with constant training baseline"
            )

        if transition_rate < 0.01:
            return (
                "Statistical deviation detector",
                "Change-point detection",
                "Point labels with very low transition rate"
            )

        return (
            "Robust statistical detector",
            "Isolation Forest",
            "Point anomaly labels with variable telemetry"
        )

    return (
        "General anomaly detector",
        "Statistical + temporal comparison",
        "Review anomaly label composition"
    )


# ============================================================
# SECTION 6: BUILD THE 82-CHANNEL DECISION MATRIX
# ============================================================

temporal_lookup = temporal.set_index("channel")
statistics_lookup = statistics.set_index("channel")

matrix_rows = []

for _, stat_row in statistics.iterrows():

    channel = stat_row["channel"]

    temporal_row = temporal_lookup.loc[channel]

    train_constant = bool(stat_row["train_constant"])

    transition_rate = float(
        temporal_row["test_transition_rate"]
    )

    if channel in annotation_info:

        info = annotation_info[channel]

        spacecraft = info["spacecraft"]
        classes = info["classes"]
        event_count = info["event_count"]

        annotation_status = "Annotated"

        if channel in duplicate_channels:
            annotation_status = (
                "Duplicate annotation - provisional"
            )

    else:

        spacecraft = "Unknown"
        classes = []
        event_count = 0
        annotation_status = "Unannotated"

    primary, secondary, rationale = choose_detector(
        train_constant,
        transition_rate,
        classes
    )

    feature_info = feature_summary.get(
        channel,
        {
            "strong_feature_count": 0,
            "max_abs_effect": np.nan,
            "max_abs_correlation": np.nan,
            "candidate_features": ""
        }
    )

    # Additional evidence flags

    if train_constant:
        baseline_note = "Constant training baseline"
    else:
        baseline_note = "Variable training baseline"

    if transition_rate < 0.01:
        temporal_note = "Very low transition rate"
    elif transition_rate >= 0.5:
        temporal_note = "High transition rate"
    else:
        temporal_note = "Moderate transition rate"

    if feature_info["strong_feature_count"] >= 3:
        feature_note = "Multivariate investigation recommended"
    else:
        feature_note = "No strong multivariate evidence by current threshold"

    matrix_rows.append({

        "channel": channel,
        "spacecraft": spacecraft,
        "annotation_status": annotation_status,
        "anomaly_classes": ", ".join(sorted(classes)),
        "event_count": event_count,

        "train_constant": train_constant,
        "train_std": stat_row["train_std"],
        "test_std": stat_row["test_std"],

        "test_transition_rate": transition_rate,
        "test_unique_values": temporal_row[
            "test_unique_values"
        ],

        "primary_detector": primary,
        "secondary_detector": secondary,

        "strong_feature_count": feature_info[
            "strong_feature_count"
        ],

        "max_abs_standardized_difference": feature_info[
            "max_abs_effect"
        ],

        "max_abs_telemetry_correlation": feature_info[
            "max_abs_correlation"
        ],

        "candidate_auxiliary_features": feature_info[
            "candidate_features"
        ],

        "baseline_observation": baseline_note,
        "temporal_observation": temporal_note,
        "feature_observation": feature_note,

        "selection_rationale": rationale

    })


# ============================================================
# SECTION 7: SAVE AND DISPLAY RESULTS
# ============================================================

matrix = pd.DataFrame(matrix_rows)

matrix = matrix.sort_values(
    by=["spacecraft", "channel"]
)

output_file = OUTPUT_DIR / "detector_selection_matrix.csv"

matrix.to_csv(output_file, index=False)

print("\n==========================================")
print("AURA DETECTOR-SELECTION MATRIX GENERATED")
print("==========================================")

print(f"Total channels: {len(matrix)}")

print("\nPrimary detector distribution:")

print(
    matrix["primary_detector"].value_counts().to_string()
)

print("\nAnnotation status:")

print(
    matrix["annotation_status"].value_counts().to_string()
)

print("\nFirst 15 channels:")

print(
    matrix[
        [
            "channel",
            "anomaly_classes",
            "primary_detector",
            "secondary_detector"
        ]
    ].head(15).to_string(index=False)
)

print(f"\nSaved to: {output_file}")
