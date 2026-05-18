import os
import shutil
from medmnist import PneumoniaMNIST

# AI Attribution: Used Gemini to explain how to implement using os/other libraries to avoid hardcoding file paths.
# Prompt: "How can I implement the os library to avoid hardcoding the file path of my MedMNIST dataset?"
# Based on Gemini's explanation and suggestion, I learned how to use os.path.join and what it does as well as shutil.copy.
print("Downloading PneumoniaMNIST...")
dataset = PneumoniaMNIST(split="train", download=True)

# 2. Finds where the file was saved
original_filepath = os.path.join(dataset.root, "pneumoniamnist.npz")
print(f"Downloaded to: {original_filepath}")

# 3. Copies it to current project directory
current_directory = os.getcwd()
new_filepath = os.path.join(current_directory, "pneumoniamnist.npz")

shutil.copy(original_filepath, new_filepath)
print(f"Successfully copied to your project folder: {new_filepath}")
