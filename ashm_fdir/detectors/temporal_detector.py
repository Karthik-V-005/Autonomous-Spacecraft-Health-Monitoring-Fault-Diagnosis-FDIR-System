
import numpy as np

from ashm_fdir.detectors.base_detector import BaseDetector
from ashm_fdir.detectors.temporal_features import (
    extract_temporal_features
)


class TemporalAnomalyDetector(BaseDetector):

    def __init__(self, window=10):
        super().__init__()

        self.window = window
        self.feature_medians_ = None
        self.feature_scales_ = None

    def fit(self, X_train):
        """
        Learn the normal temporal feature distributions
        using training telemetry only.
        """

        X_train = np.asarray(X_train, dtype=float).reshape(-1)

        if len(X_train) == 0:
            raise ValueError("Training telemetry cannot be empty.")

        if not np.all(np.isfinite(X_train)):
            raise ValueError("Training telemetry contains non-finite values.")

        features = extract_temporal_features(
            X_train,
            window=self.window
        )

        self.feature_medians_ = np.median(features, axis=0)

        mad = np.median(
            np.abs(features - self.feature_medians_),
            axis=0
        )

        # Convert MAD into a robust standard-deviation estimate.
        scales = 1.4826 * mad

        # Fall back to IQR when MAD is zero or extremely small.
        q75 = np.percentile(features, 75, axis=0)
        q25 = np.percentile(features, 25, axis=0)
        iqr_scales = (q75 - q25) / 1.349

        scales = np.where(scales > 1e-8, scales, iqr_scales)

        # Final fallback for features with no measurable spread.
        std_scales = np.std(features, axis=0)
        scales = np.where(scales > 1e-8, scales, std_scales)
        scales = np.where(scales > 1e-8, scales, 1.0)

        self.feature_scales_ = scales
        self.is_fitted = True

        return self

    def score(self, X, history=None):
        """
        Return one anomaly score per telemetry sample.

        Higher scores indicate more unusual temporal behaviour.
        """

        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before scoring.")

        features = extract_temporal_features(
            X,
            window=self.window,
            history=history
        )

        deviations = np.abs(
            features - self.feature_medians_
        ) / self.feature_scales_

        # Use the largest deviation among the three features.
        scores = np.max(deviations, axis=1)

        return scores

    def calibrate_threshold(self, validation_scores, percentile=99):
        """
        Calibrate the anomaly threshold using validation scores.
        """

        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before calibration.")

        validation_scores = np.asarray(
            validation_scores,
            dtype=float
        ).reshape(-1)

        if len(validation_scores) == 0:
            raise ValueError("Validation scores cannot be empty.")

        if not np.all(np.isfinite(validation_scores)):
            raise ValueError("Validation scores contain non-finite values.")

        if not 0 < percentile < 100:
            raise ValueError("Percentile must be between 0 and 100.")

        self.threshold = np.percentile(
            validation_scores,
            percentile
        )

        return self.threshold

    def predict(self, X, threshold=None, history=None):
        """
        Predict 1 for anomalous samples and 0 for normal samples.
        """

        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before prediction.")

        if threshold is None:
            threshold = self.threshold

        if threshold is None:
            raise ValueError("Calibrate a threshold before prediction.")

        scores = self.score(X, history=history)

        return (scores > threshold).astype(int)
