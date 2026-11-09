
from abc import ABC, abstractmethod
import numpy as np


class BaseDetector(ABC):
    """
    Common interface for all AURA anomaly detectors.

    Every detector must implement:
    1. fit()     - Learn normal behaviour from training data.
    2. score()   - Calculate anomaly scores.
    3. predict() - Convert scores into anomaly predictions.
    """

    def __init__(self):
        self.is_fitted = False
        self.threshold = None

    @abstractmethod
    def fit(self, X_train):
        """Learn the normal behaviour from training data."""
        pass

    @abstractmethod
    def score(self, X):
        """Return an anomaly score for each observation."""
        pass

    def calibrate_threshold(self, validation_scores, percentile=99):
        """
        Set the anomaly threshold using validation scores.

        Scores above this threshold are considered anomalous.
        """
        validation_scores = np.asarray(validation_scores, dtype=float)

        if validation_scores.size == 0:
            raise ValueError("Validation scores cannot be empty.")

        if not np.all(np.isfinite(validation_scores)):
            raise ValueError("Validation scores contain NaN or infinite values.")

        if not 0 < percentile < 100:
            raise ValueError("Percentile must be between 0 and 100.")

        self.threshold = float(np.percentile(validation_scores, percentile))
        return self.threshold

    def predict(self, X, threshold=None):
        """
        Convert anomaly scores into binary predictions.

        1 = anomaly
        0 = normal
        """
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before prediction.")

        active_threshold = threshold if threshold is not None else self.threshold

        if active_threshold is None:
            raise RuntimeError("Calibrate or provide a threshold before prediction.")

        scores = self.score(X)
        return (scores > active_threshold).astype(int)
