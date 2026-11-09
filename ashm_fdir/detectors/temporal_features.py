
import numpy as np


def extract_temporal_features(values, window=10, history=None):
    """
    Extract causal temporal features from a univariate telemetry stream.

    Features:
        1. Rolling mean of previous samples
        2. Rolling standard deviation of previous samples
        3. Absolute first difference

    history:
        Optional preceding telemetry samples. Useful when processing
        calibration or test data without losing the previous context.
    """

    values = np.asarray(values, dtype=float).reshape(-1)

    if len(values) == 0:
        raise ValueError("Input telemetry cannot be empty.")

    if window < 1:
        raise ValueError("Window must be at least 1.")

    if not np.all(np.isfinite(values)):
        raise ValueError("Telemetry contains NaN or infinite values.")

    if history is None:
        history = np.array([], dtype=float)
    else:
        history = np.asarray(history, dtype=float).reshape(-1)

        if not np.all(np.isfinite(history)):
            raise ValueError("History contains NaN or infinite values.")

    # Keep only the most recent samples needed for context.
    history = history[-window:]

    combined = np.concatenate((history, values))
    offset = len(history)

    features = np.zeros((len(values), 3), dtype=float)

    for i in range(len(values)):
        current_index = offset + i
        current_value = combined[current_index]

        # Previous samples only — current sample is excluded.
        start = max(0, current_index - window)
        previous = combined[start:current_index]

        if len(previous) == 0:
            previous = np.array([current_value])

        rolling_mean = np.mean(previous)
        rolling_std = np.std(previous)

        if current_index == 0:
            first_difference = 0.0
        else:
            first_difference = abs(
                current_value - combined[current_index - 1]
            )

        features[i] = [
            rolling_mean,
            rolling_std,
            first_difference
        ]

    return features
