import numpy as np
import matplotlib.pyplot as plt

data = np.load("data/test/P-1.npy")
telemetry = data[:, 0]

window_size = 10

predictions = []
actuals = []
errors = []

for i in range(len(telemetry) - window_size):

    window = telemetry[i:i + window_size]
    actual = telemetry[i + window_size]

    prediction = np.mean(window)
    error = abs(actual - prediction)

    predictions.append(prediction)
    actuals.append(actual)
    errors.append(error)

predictions = np.array(predictions)
actuals = np.array(actuals)
errors = np.array(errors)


# Each error corresponds to the NEXT timestep after its input window.
error_timesteps = np.arange(window_size, len(telemetry))

# Ground-truth anomaly interval
anomaly_start = 2149
anomaly_end = 2349

anomaly_mask = (
    (error_timesteps >= anomaly_start) &
    (error_timesteps <= anomaly_end)
)

normal_errors = errors[~anomaly_mask]
anomaly_errors = errors[anomaly_mask]

print("\n--- Error comparison ---")
print("Mean error outside anomaly:", np.mean(normal_errors))
print("Mean error inside anomaly:", np.mean(anomaly_errors))
print("Maximum error outside anomaly:", np.max(normal_errors))
print("Maximum error inside anomaly:", np.max(anomaly_errors))

# Plot with the error values aligned to their actual target timesteps
plt.figure(figsize=(14, 5))
plt.plot(error_timesteps, errors, linewidth=0.8)
plt.axvspan(
    anomaly_start,
    anomaly_end,
    alpha=0.25,
    color="orange",
    label="Ground-truth anomaly"
)

plt.xlabel("Actual target timestep")
plt.ylabel("Absolute prediction error")
plt.title("P-1: Prediction Error vs Ground-Truth Anomaly")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


print("Number of predictions:", len(predictions))
print("Mean error:", np.mean(errors))
print("Maximum error:", np.max(errors))

# -------------------------
# Plot prediction error
# -------------------------

plt.figure(figsize=(14, 5))

plt.plot(errors)

# Ground-truth anomaly
plt.axvspan(
    2149,
    2349,
    alpha=0.25,
    label="Ground-truth anomaly"
)

plt.xlabel("Timestep")
plt.ylabel("Prediction error")
plt.title("P-1 — Simple Moving-Average Prediction Error")
plt.legend()
plt.grid(True)

plt.show()