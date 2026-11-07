import numpy as np

test_data = np.load("../data/test/G-7.npy")
telemetry = test_data[:, 0]

anomalies = [
    (3650, 3750, "A1 Contextual"),
    (5050, 5100, "A2 Point"),
    (7560, 7675, "A3 Contextual")
]


def inspect_runs(values, start, end, name):

    print("\n" + "=" * 45)
    print(name)
    print("=" * 45)

    segment = values[start:end + 1]

    for target in [-1.0, -0.2, 0.6]:

        mask = np.isclose(segment, target, atol=1e-6)

        indices = np.where(mask)[0] + start

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

        print(f"\nValue: {target}")
        print("Occurrences:", len(indices))
        print("Consecutive runs:", runs)


for start, end, name in anomalies:

    inspect_runs(
        telemetry,
        start,
        end,
        name
    )