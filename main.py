import numpy as np
import matplotlib.pyplot as plt
from mlp_classifier import MultiLayerPerceptron, load_and_preprocess_data


def calculate_accuracy(predictions, labels):
    """Converts raw probability scores into 0 or 1 guesses, then compares to true labels."""
    rounded_guesses = np.round(predictions)
    correct_count = np.sum(rounded_guesses == labels)
    total_samples = labels.shape[0]
    return (correct_count / total_samples) * 100.0


if __name__ == "__main__":
    # 1. Load Data
    print("Loading PneumoniaMNIST data...")
    X_train, y_train, X_test, y_test = load_and_preprocess_data("pneumoniamnist.npz")

    # 2. Build the Network Architecture
    print("Initializing Multi-Layer Perceptron (784 -> 128 -> 64 -> 1)...")
    model = MultiLayerPerceptron(
        input_nodes=784, hidden_nodes_1=128, hidden_nodes_2=64, output_nodes=1
    )

    # 3. Train the Network and Save the History
    print("Starting Training Loop...")

    # We now capture the returned list into a variable called 'history'
    history = model.train(X_train, y_train, epochs=10000, learning_rate=0.01)

    # 4. Plot the Learning Curve
    print("Generating Loss Graph...")
    plt.figure(figsize=(8, 5))
    plt.plot(history, label="Training Loss", color="blue", linewidth=2)
    plt.title("MLP Training Loss over Epochs (Pneumonia Detection)")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error Loss")
    plt.legend()
    plt.grid(True)
    plt.show()  # This will pop open a window showing your graph!

    # 5. Final Exam (Evaluate on the Test Set)
    print("\nAdministering Final Exam on Test Data...")
    test_predictions = model.feed_forward(X_test)
    final_accuracy = calculate_accuracy(test_predictions, y_test)

    print(f"Final Test Accuracy: {final_accuracy:.2f}%")
