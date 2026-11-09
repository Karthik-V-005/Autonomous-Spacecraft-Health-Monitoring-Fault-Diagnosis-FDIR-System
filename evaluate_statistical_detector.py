
import ast
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

from ashm_fdir.data_loader import load_all_channels
from ashm_fdir.detectors.statistical_detector import (
    RobustStatisticalDetector
)


# --------------------------------------------------
# 1. PROJECT PATHS AND CONFIGURATION
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

ANNOTATION_FILE = PROJECT_DIR / "labeled_anomalies.csv"

OUTPUT_DIR = PROJECT_DIR / "statistical_detector_models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTION_DIR = OUTPUT_DIR / "predictions"
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_SPLIT = 0.80
THRESHOLD_PERCENTILE = 99


# --------------------------------------------------
# 2. LOAD ANNOTATIONS
# --------------------------------------------------

def load_annotations():

    annotations = pd.read_csv(ANNOTATION_FILE)

    required_columns = {
        "chan_id",
        "anomaly_sequences",
        "num_values"
    }

    missing = required_columns - set(annotations.columns)

    if missing:
        raise ValueError(
            f"Missing annotation columns: {missing}"
        )

    # Identify duplicate channel annotations.
    duplicate_channels = annotations[
        annotations["chan_id"].duplicated(keep=False)
    ]["chan_id"].unique()

    if len(duplicate_channels) > 0:
        print("\nWARNING: Duplicate annotation records found:")
        print(duplicate_channels)

    # Exclude ambiguous duplicates instead of silently
    # choosing one of their conflicting annotation records.
    annotations = annotations[
        ~annotations["chan_id"].isin(duplicate_channels)
    ]

    annotation_map = {}

    for _, row in annotations.iterrows():

        channel = row["chan_id"]

        sequences = ast.literal_eval(
            str(row["anomaly_sequences"])
        )

        annotation_map[channel] = {
            "sequences": sequences,
            "num_values": int(row["num_values"])
        }

    return annotation_map


# --------------------------------------------------
# 3. CONVERT ANOMALY INTERVALS INTO BINARY LABELS
# --------------------------------------------------

def create_ground_truth(sequences, data_length):

    labels = np.zeros(data_length, dtype=int)

    for start, end in sequences:

        start = int(start)
        end = int(end)

        if start < 0 or end >= data_length or start > end:
            raise ValueError(
                f"Invalid anomaly interval: [{start}, {end}]"
            )

        # Both interval endpoints are inclusive.
        labels[start:end + 1] = 1

    return labels


# --------------------------------------------------
# 4. EVALUATE ONE CHANNEL
# --------------------------------------------------

def evaluate_channel(channel, data, annotation):

    train = np.asarray(
        data["train_telemetry"],
        dtype=float
    )

    test = np.asarray(
        data["test_telemetry"],
        dtype=float
    )

    if len(test) != annotation["num_values"]:
        raise ValueError(
            f"{channel}: Test length does not match annotation."
        )

    if not np.all(np.isfinite(train)):
        raise ValueError(
            f"{channel}: Training data contains non-finite values."
        )

    if not np.all(np.isfinite(test)):
        raise ValueError(
            f"{channel}: Test data contains non-finite values."
        )

    # Chronological training split.
    split_index = int(len(train) * TRAIN_SPLIT)

    if split_index == 0 or split_index == len(train):
        raise ValueError(
            f"{channel}: Insufficient training observations."
        )

    fit_data = train[:split_index]
    calibration_data = train[split_index:]

    # Fit detector using the first 80%.
    detector = RobustStatisticalDetector()
    detector.fit(fit_data)

    # Calibrate threshold using the final 20%.
    calibration_scores = detector.score(
        calibration_data
    )

    threshold = detector.calibrate_threshold(
        calibration_scores,
        percentile=THRESHOLD_PERCENTILE
    )

    # Score the complete test telemetry.
    test_scores = detector.score(test)

    predictions = detector.predict(
        test,
        threshold=threshold
    )

    # Convert labelled intervals into sample-level labels.
    ground_truth = create_ground_truth(
        annotation["sequences"],
        len(test)
    )

    # Calculate evaluation metrics.
    tn, fp, fn, tp = confusion_matrix(
        ground_truth,
        predictions,
        labels=[0, 1]
    ).ravel()

    precision = precision_score(
        ground_truth,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        ground_truth,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        ground_truth,
        predictions,
        zero_division=0
    )

    if len(np.unique(ground_truth)) == 2:

        roc_auc = roc_auc_score(
            ground_truth,
            test_scores
        )

        average_precision = average_precision_score(
            ground_truth,
            test_scores
        )

    else:
        roc_auc = np.nan
        average_precision = np.nan

    # Save individual channel predictions.
    prediction_df = pd.DataFrame({
        "timestep": np.arange(len(test)),
        "actual": ground_truth,
        "anomaly_score": test_scores,
        "predicted": predictions
    })

    prediction_df.to_csv(
        PREDICTION_DIR / f"{channel}_predictions.csv",
        index=False
    )

    return {
        "channel": channel,
        "train_samples": len(train),
        "test_samples": len(test),
        "fit_samples": len(fit_data),
        "calibration_samples": len(calibration_data),
        "threshold": threshold,
        "actual_anomalies": int(ground_truth.sum()),
        "predicted_anomalies": int(predictions.sum()),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "training_constant": detector.is_constant_
    }


# --------------------------------------------------
# 5. MAIN EVALUATION PIPELINE
# --------------------------------------------------

def main():

    print("=" * 60)
    print("AURA - ROBUST STATISTICAL DETECTOR")
    print("=" * 60)

    print("\nLoading telemetry channels...")

    matrix, channel_data, failures = load_all_channels()

    print("Loaded channels:", len(channel_data))
    print("Loader failures:", len(failures))

    print("\nLoading anomaly annotations...")

    annotation_map = load_annotations()

    print(
        "Unique, unambiguous annotated channels:",
        len(annotation_map)
    )

    results = []
    skipped = []

    print("\nStarting channel evaluation...\n")

    for channel, data in channel_data.items():

        if channel not in annotation_map:

            print(f"[SKIPPED] {channel}: No usable annotation.")
            skipped.append(channel)
            continue

        try:

            result = evaluate_channel(
                channel,
                data,
                annotation_map[channel]
            )

            results.append(result)

            print(
                f"[DONE] {channel} | "
                f"F1: {result['f1_score']:.4f} | "
                f"Precision: {result['precision']:.4f} | "
                f"Recall: {result['recall']:.4f}"
            )

        except Exception as error:

            print(f"[ERROR] {channel}: {error}")
            skipped.append(channel)

    # --------------------------------------------------
    # 6. SAVE RESULTS
    # --------------------------------------------------

    results_df = pd.DataFrame(results)

    output_file = (
        OUTPUT_DIR / "robust_statistical_evaluation.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    print("Successfully evaluated:", len(results))
    print("Skipped channels:", len(skipped))

    if not results_df.empty:

        print("\nOverall metric averages:")

        print(
            results_df[
                [
                    "precision",
                    "recall",
                    "f1_score",
                    "roc_auc",
                    "average_precision"
                ]
            ].mean().round(4)
        )

        print("\nTop 10 channels by F1-score:")

        print(
            results_df.sort_values(
                "f1_score",
                ascending=False
            )[
                [
                    "channel",
                    "precision",
                    "recall",
                    "f1_score",
                    "roc_auc"
                ]
            ].head(10).to_string(index=False)
        )

    print("\nResults saved to:")
    print(output_file)

    print("\nIndividual predictions saved to:")
    print(PREDICTION_DIR)

    if skipped:
        print("\nSkipped channels:")
        print(skipped)


if __name__ == "__main__":
    main()
