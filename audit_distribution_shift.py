
from pathlib import Path

import numpy as np
import pandas as pd

from ashm_fdir.data_loader import load_all_channels
from ashm_fdir.detectors.statistical_detector import (
    RobustStatisticalDetector
)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

OUTPUT_DIR = ROOT / "statistical_detector_models"

AUDIT_FILE = OUTPUT_DIR / "distribution_shift_audit.csv"

CHANNELS = ["A-1", "A-8", "A-9", "D-4", "P-1"]

TRAIN_SPLIT = 0.80
PERCENTILE = 99


# --------------------------------------------------
# STATISTICS HELPER
# --------------------------------------------------

def describe_signal(values):

    return {
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95))
    }


# --------------------------------------------------
# AUDIT ONE CHANNEL
# --------------------------------------------------

def audit_channel(channel, data, prediction_df):

    train = data["train_telemetry"]
    test = data["test_telemetry"]

    actual = prediction_df["actual"].to_numpy()

    split = int(len(train) * TRAIN_SPLIT)

    fit_data = train[:split]
    calibration_data = train[split:]

    detector = RobustStatisticalDetector()
    detector.fit(fit_data)

    calibration_scores = detector.score(calibration_data)
    test_scores = detector.score(test)

    threshold = detector.calibrate_threshold(
        calibration_scores,
        percentile=PERCENTILE
    )

    segments = {
        "fit": fit_data,
        "calibration": calibration_data,
        "test_normal": test[actual == 0],
        "test_anomaly": test[actual == 1]
    }

    score_segments = {
        "fit": detector.score(fit_data),
        "calibration": calibration_scores,
        "test_normal": test_scores[actual == 0],
        "test_anomaly": test_scores[actual == 1]
    }

    result = {
        "channel": channel,
        "training_median": detector.median_,
        "training_MAD": detector.mad_,
        "training_scale": detector.scale_,
        "training_constant": detector.is_constant_,
        "threshold": threshold
    }

    print("\n" + "=" * 65)
    print(f"CHANNEL: {channel}")
    print("=" * 65)

    print(f"Training baseline median: {detector.median_:.6f}")
    print(f"Training MAD: {detector.mad_:.6f}")
    print(f"Training scale: {detector.scale_:.6f}")
    print(f"Calibration threshold: {threshold:.6f}")

    for name in segments:

        raw_stats = describe_signal(segments[name])
        score_stats = describe_signal(score_segments[name])

        print(f"\n{name.upper()}")

        print("Raw telemetry:", raw_stats)
        print("Anomaly scores:", score_stats)

        for key, value in raw_stats.items():
            result[f"{name}_raw_{key}"] = value

        for key, value in score_stats.items():
            result[f"{name}_score_{key}"] = value

    return result


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("Loading telemetry...")

    matrix, channel_data, failures = load_all_channels()

    print("Loaded:", len(channel_data))
    print("Failures:", len(failures))

    results = []

    for channel in CHANNELS:

        if channel not in channel_data:
            print(f"[SKIP] {channel}: Not found.")
            continue

        prediction_file = (
            OUTPUT_DIR / "predictions" /
            f"{channel}_predictions.csv"
        )

        if not prediction_file.exists():
            print(f"[SKIP] {channel}: Predictions missing.")
            continue

        prediction_df = pd.read_csv(prediction_file)

        result = audit_channel(
            channel,
            channel_data[channel],
            prediction_df
        )

        results.append(result)

    if results:

        output_df = pd.DataFrame(results)

        output_df.to_csv(
            AUDIT_FILE,
            index=False
        )

        print("\nDistribution audit saved:")
        print(AUDIT_FILE)


if __name__ == "__main__":
    main()
