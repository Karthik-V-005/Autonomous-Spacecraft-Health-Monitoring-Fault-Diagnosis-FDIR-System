
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, average_precision_score


# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

RESULTS_DIR = ROOT / "statistical_detector_models"

PREDICTION_DIR = RESULTS_DIR / "predictions"

SUMMARY_FILE = RESULTS_DIR / "robust_statistical_evaluation.csv"

OUTPUT_FILE = RESULTS_DIR / "statistical_audit.csv"

CHANNELS = ["A-1", "A-8", "A-9", "D-4", "P-1"]


# --------------------------------------------------
# AUDIT ONE CHANNEL
# --------------------------------------------------

def audit_channel(channel, summary):

    file_path = PREDICTION_DIR / f"{channel}_predictions.csv"

    if not file_path.exists():
        print(f"[MISSING] {channel}: Prediction file not found.")
        return None

    data = pd.read_csv(file_path)

    actual = data["actual"].to_numpy(dtype=int)
    scores = data["anomaly_score"].to_numpy(dtype=float)
    predicted = data["predicted"].to_numpy(dtype=int)

    threshold = float(summary["threshold"])

    normal_scores = scores[actual == 0]
    anomaly_scores = scores[actual == 1]

    if len(normal_scores) == 0 or len(anomaly_scores) == 0:
        print(f"[INVALID] {channel}: Only one ground-truth class.")
        return None

    auc = roc_auc_score(actual, scores)
    inverted_auc = roc_auc_score(actual, -scores)

    ap = average_precision_score(actual, scores)

    result = {
        "channel": channel,
        "threshold": threshold,
        "training_constant": summary["training_constant"],
        "normal_score_median": np.median(normal_scores),
        "anomaly_score_median": np.median(anomaly_scores),
        "normal_score_p95": np.percentile(normal_scores, 95),
        "anomaly_score_p05": np.percentile(anomaly_scores, 5),
        "score_auc": auc,
        "inverted_score_auc": inverted_auc,
        "average_precision": ap,
        "predicted_anomaly_fraction": predicted.mean(),
        "actual_anomaly_fraction": actual.mean(),
        "score_min": scores.min(),
        "score_max": scores.max()
    }

    print("\n" + "=" * 55)
    print(f"CHANNEL: {channel}")
    print("=" * 55)

    for key, value in result.items():

        if key != "channel":
            print(f"{key}: {value}")

    if threshold <= 1e-12:
        print("\nWARNING: Threshold is zero or extremely small.")

    if inverted_auc > auc:
        print("\nWARNING: Inverted scores rank anomalies better.")

    if predicted.mean() > 0.90:
        print("\nWARNING: More than 90% of test samples are predicted anomalous.")

    if predicted.mean() < 0.001:
        print("\nWARNING: Almost no test samples are predicted anomalous.")

    return result


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    summary_df = pd.read_csv(SUMMARY_FILE)

    summary_df = summary_df.set_index("channel")

    results = []

    for channel in CHANNELS:

        if channel not in summary_df.index:
            print(f"[MISSING] {channel}: No summary record.")
            continue

        result = audit_channel(
            channel,
            summary_df.loc[channel]
        )

        if result is not None:
            results.append(result)

    if results:

        audit_df = pd.DataFrame(results)

        audit_df.to_csv(
            OUTPUT_FILE,
            index=False
        )

        print("\nAudit saved to:")
        print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
