
import numpy as np


# ============================================================
# SECTION 1: ROBUST SCALER
# ============================================================

class RobustScaler:

    def __init__(self):
        self.median_ = None
        self.scale_ = None
        self.constant_mask_ = None

    def fit(self, data):

        data = np.asarray(data, dtype=np.float64)

        if data.ndim != 2:
            raise ValueError("Expected a 2D feature array.")

        if not np.isfinite(data).all():
            raise ValueError(
                "Training data contains NaN or infinite values."
            )

        self.median_ = np.median(data, axis=0)

        q75 = np.percentile(data, 75, axis=0)
        q25 = np.percentile(data, 25, axis=0)

        iqr = q75 - q25

        self.constant_mask_ = (
            np.ptp(data, axis=0) == 0
        )

        # Avoid division by zero.
        self.scale_ = np.where(iqr > 1e-12, iqr, 1.0)

        return self

    def transform(self, data):

        if self.median_ is None:
            raise RuntimeError("Scaler has not been fitted.")

        data = np.asarray(data, dtype=np.float64)

        if data.ndim != 2:
            raise ValueError("Expected a 2D feature array.")

        if data.shape[1] != len(self.median_):
            raise ValueError("Feature count differs from training data.")

        if not np.isfinite(data).all():
            raise ValueError(
                "Input data contains NaN or infinite values."
            )

        return (data - self.median_) / self.scale_

    def fit_transform(self, data):

        self.fit(data)
        return self.transform(data)


# ============================================================
# SECTION 2: CHANNEL PREPROCESSOR
# ============================================================

class ChannelPreprocessor:

    def __init__(self):

        self.telemetry_scaler = RobustScaler()
        self.auxiliary_scaler = RobustScaler()

        self.has_auxiliary = False
        self.fitted = False

    def fit(self, train_data):

        train_data = np.asarray(
            train_data,
            dtype=np.float64
        )

        if train_data.ndim != 2:
            raise ValueError("Expected 2D training data.")

        if train_data.shape[0] == 0:
            raise ValueError("Training data is empty.")

        if not np.isfinite(train_data).all():
            raise ValueError(
                "Training data contains invalid values."
            )

        telemetry = train_data[:, [0]]

        self.telemetry_scaler.fit(telemetry)

        auxiliary = train_data[:, 1:]

        self.has_auxiliary = auxiliary.shape[1] > 0

        if self.has_auxiliary:
            self.auxiliary_scaler.fit(auxiliary)

        self.fitted = True

        return self

    def transform(self, data):

        if not self.fitted:
            raise RuntimeError(
                "Preprocessor must be fitted first."
            )

        data = np.asarray(
            data,
            dtype=np.float64
        )

        if data.ndim != 2:
            raise ValueError("Expected 2D input data.")

        telemetry = self.telemetry_scaler.transform(
            data[:, [0]]
        )

        auxiliary = data[:, 1:]

        if self.has_auxiliary:
            auxiliary = self.auxiliary_scaler.transform(
                auxiliary
            )

            return np.column_stack([
                telemetry,
                auxiliary
            ])

        return telemetry

    def fit_transform(self, train_data):

        self.fit(train_data)

        return self.transform(train_data)


# ============================================================
# SECTION 3: PREPROCESSING SUMMARY
# ============================================================

def preprocessing_summary(preprocessor):

    telemetry_constant = bool(
        preprocessor.telemetry_scaler.constant_mask_[0]
    )

    auxiliary_constant_count = (
        int(
            np.sum(
                preprocessor.auxiliary_scaler.constant_mask_
            )
        )
        if preprocessor.has_auxiliary
        else 0
    )

    return {
        "telemetry_constant": telemetry_constant,
        "auxiliary_constant_features": auxiliary_constant_count,
        "auxiliary_feature_count": (
            len(preprocessor.auxiliary_scaler.median_)
            if preprocessor.has_auxiliary
            else 0
        )
    }
