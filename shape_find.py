import numpy as np

cell_file = np.load("pneumoniamnist.npz")
print("Keys inside this file:")
print(cell_file.files)
print("-" * 30)


for key in cell_file.files:
    data_array = cell_file[key]

    print(f"Array Name: {key}")
    print(f"  Shape: {data_array.shape}")
    print(f"  Data Type: {data_array.dtype}")
    print(f"  Min Value: {np.min(data_array)}")
    print(f"  Max Value: {np.max(data_array)}")
    print("-" * 30)
