
import numpy as np
import pandas as pd
from pathlib import Path

from ashm_fdir.data_loader import load_all_channels
from ashm_fdir.preprocessing import (
    ChannelPreprocessor,
    preprocessing_summary
)


# ============================================================
# SECTION 1: CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = PROJECT_DIR / "preprocessing_results"
PARAMETER_DIR = OUTPUT_DIR / "scaler_parameters"

OUTPUT_DIR.mkdir(exist_ok=True)
PARAMETER_DIR.mkdir(exist_ok=True)


# ============================================================
# SECTION 2: LOAD ALL CHANNELS
# ============================================================

matrix, channel_data, failures = load_all_channels()

print("\n========================================")
print("AURA FULL PREPROCESSING VALIDATION")
print("========================================")

print(f"Channels discovered: {len(matrix)}")
print(f"Channels loaded: {len(channel_data)}")


# ============================================================
# SECTION 3: PREPROCESS EACH CHANNEL
# ============================================================

results = []
preprocessing_failures = []

for channel, data in channel_data.items():

    print(f"\nProcessing {channel}...")

    try:

        train_data = data["train_data"]
        test_data = data["test_data"]

        preprocessor = ChannelPreprocessor()

        # Fit only on training data.
        train_scaled = preprocessor.fit_transform(train_data)

        # Apply the fitted scaler to test data.
        test_scaled = preprocessor.transform(test_data)

        summary = preprocessing_summary(preprocessor)

        # ----------------------------------------------------
        # TELEMETRY DISTRIBUTION INFORMATION
        # ----------------------------------------------------

        telemetry_scaler = preprocessor.telemetry_scaler

        telemetry_iqr = float(
            np.percentile(train_data[:, 0], 75)
            - np.percentile(train_data[:, 0], 25)
        )

        telemetry_zero_iqr = (
            np.percentile(train_data[:, 0], 75)
            - np.percentile(train_data[:, 0], 25)
        ) <= 1e-12

        # ----------------------------------------------------
        # AUXILIARY FEATURE INFORMATION
        # ----------------------------------------------------

        if preprocessor.has_auxiliary:

            auxiliary_scaler = preprocessor.auxiliary_scaler

            constant_features = int(
                np.sum(auxiliary_scaler.constant_mask_)
            )

            variable_features = (
                train_data.shape[1] - 1 - constant_features
            )

            nonzero_iqr_features = int(
                np.sum(
                    (auxiliary_scaler.scale_ > 1e-12)
                    & (~auxiliary_scaler.constant_mask_)
                )
            )

            zero_iqr_features = (
                train_data.shape[1] - 1 - nonzero_iqr_features
            )            

        else:

            constant_features = 0
            variable_features = 0
            nonzero_iqr_features = 0
            zero_iqr_features = 0

        # ----------------------------------------------------
        # VALIDATION CHECKS
        # ----------------------------------------------------

        assert train_scaled.shape == train_data.shape
        assert test_scaled.shape == test_data.shape

        assert np.isfinite(train_scaled).all()
        assert np.isfinite(test_scaled).all()

        # ----------------------------------------------------
        # SAVE FITTED PARAMETERS
        # ----------------------------------------------------

        parameter_path = (
            PARAMETER_DIR / f"{channel}_scaler.npz"
        )

        np.savez_compressed(
            parameter_path,

            telemetry_median=telemetry_scaler.median_,
            telemetry_scale=telemetry_scaler.scale_,
            telemetry_constant_mask=telemetry_scaler.constant_mask_,

            auxiliary_median=(
                preprocessor.auxiliary_scaler.median_
                if preprocessor.has_auxiliary
                else np.array([])
            ),

            auxiliary_scale=(
                preprocessor.auxiliary_scaler.scale_
                if preprocessor.has_auxiliary
                else np.array([])
            ),

            auxiliary_constant_mask=(
                preprocessor.auxiliary_scaler.constant_mask_
                if preprocessor.has_auxiliary
                else np.array([], dtype=bool)
            )
        )

        # ----------------------------------------------------
        # RECORD CHANNEL RESULTS
        # ----------------------------------------------------

        results.append({

            "channel": channel,

            "train_timesteps": train_data.shape[0],
            "test_timesteps": test_data.shape[0],

            "total_features": train_data.shape[1],
            "auxiliary_features": train_data.shape[1] - 1,

            "telemetry_constant": summary[
                "telemetry_constant"
            ],

            "telemetry_zero_iqr": telemetry_zero_iqr,

            "telemetry_training_iqr": telemetry_iqr,

            "constant_auxiliary_features": constant_features,

            "variable_auxiliary_features": variable_features,

            "nonzero_iqr_auxiliary_features": nonzero_iqr_features,

            "zero_iqr_auxiliary_features": zero_iqr_features,

            "train_scaled_shape": str(train_scaled.shape),
            "test_scaled_shape": str(test_scaled.shape),

            "validation": "PASSED"

        })

        print("  Preprocessing: PASSED")
        print(f"  Constant telemetry: {summary['telemetry_constant']}")
        print(f"  Constant auxiliary features: {constant_features}")
        print(f"  Nonzero-IQR auxiliary features: {nonzero_iqr_features}")

    except Exception as error:

        preprocessing_failures.append({
            "channel": channel,
            "error": str(error)
        })

        print(f"  FAILED: {error}")


# ============================================================
# SECTION 4: SAVE REPORTS
# ============================================================

results_df = pd.DataFrame(results)

report_path = OUTPUT_DIR / "preprocessing_validation.csv"

results_df.to_csv(report_path, index=False)

if preprocessing_failures:

    failures_df = pd.DataFrame(preprocessing_failures)

    failures_path = (
        OUTPUT_DIR / "preprocessing_failures.csv"
    )

    failures_df.to_csv(failures_path, index=False)


# ============================================================
# SECTION 5: FINAL SUMMARY
# ============================================================

print("\n========================================")
print("AURA PREPROCESSING SUMMARY")
print("========================================")

print(f"Channels loaded: {len(channel_data)}")
print(f"Preprocessing passed: {len(results)}")
print(f"Preprocessing failed: {len(preprocessing_failures)}")

if len(results_df) > 0:

    print(
        "\nChannels with constant telemetry:",
        int(results_df["telemetry_constant"].sum())
    )

    print(
        "Channels with zero telemetry IQR:",
        int(results_df["telemetry_zero_iqr"].sum())
    )

    print(
        "Total constant auxiliary features:",
        int(results_df["constant_auxiliary_features"].sum())
    )

    print(
        "Total variable auxiliary features:",
        int(results_df["variable_auxiliary_features"].sum())
    )

    print(
        "Total zero-IQR auxiliary features:",
        int(results_df["zero_iqr_auxiliary_features"].sum())
    )

print("\nValidation report:")
print(report_path)

print("\nScaler parameters:")
print(PARAMETER_DIR)

print("\n========================================")
print("FULL PREPROCESSING VALIDATION COMPLETE")
print("========================================")
