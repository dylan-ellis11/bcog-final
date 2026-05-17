import numpy as np
import matplotlib.pyplot as plt
from mlp_classifier import MultiLayerPerceptron, load_and_preprocess_data


def calculate_accuracy(predictions, labels):
    # Round decimal predictions to nearest integer to create binary guess.
    rounded_guesses = np.round(predictions)
    # Count number of correct guesses by comparing to true data labels.
    correct_count = np.sum(rounded_guesses == labels)
    # Give percentage of correct guesses out of total number of samples.
    return (correct_count / labels.shape[0]) * 100.0


if __name__ == "__main__":
    print("Loading PneumoniaMNIST data...")
    # Unpack all 6 arrays of data.
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data(
        "pneumoniamnist.npz"
    )

    # Initialization of neural network architecture.
    print("Initializing Multi-Layer Perceptron (784 -> 128 -> 64 -> 1)...")
    model = MultiLayerPerceptron(
        input_nodes=784, hidden_nodes_1=128, hidden_nodes_2=64, output_nodes=1
    )

    print("Starting Training Loop...")
    # Capture both histories
    train_history, val_history = model.train(
        X_train, y_train, X_val, y_val, epochs=2500, learning_rate=0.01
    )

    print("Generating Loss Graph...")
    plt.figure(figsize=(10, 6))

    # Plot both lines of loss values side-by-side
    plt.plot(
        train_history, label="Training Loss (Memorization)", color="blue", linewidth=2
    )
    plt.plot(
        val_history,
        label="Validation Loss (Actual Generalization)",
        color="orange",
        linewidth=2,
    )

    plt.title("Training vs. Validation Loss (Detecting Overfitting)")
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\n Evaluating on Test Data")
    test_predictions = model.feed_forward(X_test)
    final_accuracy = calculate_accuracy(test_predictions, y_test)
    print(f"Final Test Accuracy: {final_accuracy:.2f}%")
