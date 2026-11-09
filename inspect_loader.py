from ashm_fdir.data_loader import load_all_channels

matrix, channel_data, failures = load_all_channels()

print("\nMATRIX COLUMNS:")
print(matrix.columns.tolist())

print("\nCHANNEL DATA KEYS:")
first_channel = next(iter(channel_data))
print("Channel:", first_channel)
print(channel_data[first_channel].keys())

print("\nARRAY SHAPES:")
for key, value in channel_data[first_channel].items():
    if hasattr(value, "shape"):
        print(key, value.shape)

print("\nFAILURES:", len(failures))