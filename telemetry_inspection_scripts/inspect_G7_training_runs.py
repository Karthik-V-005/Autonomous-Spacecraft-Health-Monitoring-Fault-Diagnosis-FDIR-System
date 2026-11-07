import numpy as np

train_data = np.load("../data/train/G-7.npy")
telemetry = train_data[:, 0]

for target in [-1.0, 0.333]:

    mask = np.isclose(
        telemetry,
        target,
        atol=1e-6
    )

    indices = np.where(mask)[0]

    runs = []

    if len(indices) > 0:

        run_start = indices[0]
        previous = indices[0]

        for index in indices[1:]:

            if index == previous + 1:
                previous = index

            else:
                runs.append((run_start, previous))
                run_start = index
                previous = index

        runs.append((run_start, previous))

    print("\nValue:", target)
    print("Occurrences:", len(indices))
    print("Consecutive runs:", runs)