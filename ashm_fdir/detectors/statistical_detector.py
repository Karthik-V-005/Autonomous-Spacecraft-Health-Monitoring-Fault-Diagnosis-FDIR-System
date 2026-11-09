
import numpy as np

from ashm_fdir.detectors.base_detector import BaseDetector


class RobustStatisticalDetector(BaseDetector):
    """
    Detects deviations from the training telemetry baseline
    using the median and Median Absolute Deviation (MAD).
    """

    def __init__(self, epsilon=1e-12):
        super().__init__()
        self.epsilon = epsilon
        self.median_ = None
        self.mad_ = None
        self.scale_ = None
        self.is_constant_ = False

    def fit(self, X_train):
        X_train = np.asarray(X_train, dtype=float).reshape(-1)

        if X_train.size == 0:
            raise ValueError("Training data cannot be empty.")

        if not np.all(np.isfinite(X_train)):
            raise ValueError("Training data contains NaN or infinite values.")

        self.median_ = float(np.median(X_train))
        self.mad_ = float(np.median(np.abs(X_train - self.median_)))

        self.is_constant_ = bool(np.ptp(X_train) <= self.epsilon)

        # Convert MAD into a standard-deviation-like scale.
        self.scale_ = 1.4826 * self.mad_

        # Fallback for non-constant signals with zero MAD.
        if self.scale_ <= self.epsilon and not self.is_constant_:
            q75, q25 = np.percentile(X_train, [75, 25])
            self.scale_ = float((q75 - q25) / 1.349)

        # If training data is genuinely constant, use raw deviation.
        if self.is_constant_:
            self.scale_ = 1.0

        self.is_fitted = True
        return self

    def score(self, X):
        if not self.is_fitted:
            raise RuntimeError("Fit the detector before scoring.")

        X = np.asarray(X, dtype=float).reshape(-1)

        if not np.all(np.isfinite(X)):
            raise ValueError("Input data contains NaN or infinite values.")

        deviations = np.abs(X - self.median_)

        if self.is_constant_:
            return deviations

        return deviations / max(self.scale_, self.epsilon)
