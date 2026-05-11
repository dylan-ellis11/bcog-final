import os
import shutil
from medmnist import PneumoniaMNIST

# 1. This triggers the download
print("Downloading PneumoniaMNIST...")
dataset = PneumoniaMNIST(split="train", download=True)

# 2. Finds where the file was saved
original_filepath = os.path.join(dataset.root, "pneumoniamnist.npz")
print(f"Downloaded to: {original_filepath}")

# 3. Copies it directly to current project directory
current_directory = os.getcwd()
new_filepath = os.path.join(current_directory, "pneumoniamnist.npz")

shutil.copy(original_filepath, new_filepath)
print(f"Successfully copied to your project folder: {new_filepath}")
