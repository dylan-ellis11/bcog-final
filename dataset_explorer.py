import numpy as np

# Load the file
data = np.load("pneumoniamnist.npz")
train_labels = data["train_labels"]

# Find the unique labels and count how many of each exist
unique_classes, counts = np.unique(train_labels, return_counts=True)

print("PneumoniaMNIST Class Distribution in Training Data:")
print("-" * 45)
for cls, count in zip(unique_classes, counts):
    print(f"Class {cls}: {count} images")
