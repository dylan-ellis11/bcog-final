# mlp_classifier.py outline

import numpy as np


def load_and_preprocess_data(filepath):
    """
    (Corresponds to user's 'function_loading')
    Loads raw image data from an .npz file or directory.
    - Converts to grayscale (if RGB).
    - Resizes to uniform pixel dimensions (28x28).
    - Flattens 2D arrays into 1D arrays (size 784).
    - Normalizes pixel values by dividing by 255.0.
    Returns: X_train, y_train, X_test, y_test
    """
    data = np.load(filepath)

    X_train = data["train_images"]
    y_train = data["train_labels"]
    X_test = data["test_images"]
    y_test = data["test_labels"]

    # Flatten: Reshape from (N, 28, 28) to (N, 784)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # Normalize: Convert from 0-255 to 0.0-1.0
    X_train = X_train.astype(np.float64) / 255.0
    X_test = X_test.astype(np.float64) / 255.0

    return X_train, y_train, X_test, y_test


class MultiLayerPerceptron:
    def __init__(self, input_nodes, hidden_nodes_1, hidden_nodes_2, output_nodes):
        """
        Initializes the 3-layer network architecture.
        Sets up the weight matrices and bias vectors with random small values.
        """
        # initialize weight matrices
        self.input_nodes = input_nodes
        # Input to hidden layer 1
        self.W1 = np.random.randn(input_nodes, hidden_nodes_1) * 0.1
        self.b1 = np.zeros((1, hidden_nodes_1))
        # Hidden layer 1 to hidden layer 2
        self.W2 = np.random.randn(hidden_nodes_1, hidden_nodes_2) * 0.1
        self.b2 = np.zeros((1, hidden_nodes_2))
        # Hidden layer 2 to output layer
        self.W3 = np.random.randn(hidden_nodes_2, output_nodes) * 0.1
        self.b3 = np.zeros((1, output_nodes))

    def _sigmoid(self, x):
        """Activation function for the layers."""
        x = np.clip(x, -300, 300)
        return 1.0 / (1.0 + np.exp(-x))

    def _sigmoid_derivative(self, sigmoid_output):
        """Calculates the derivative for backpropagation calculus."""
        return sigmoid_output * (1.0 - sigmoid_output)

    def feed_forward(self, X):
        """
        (Corresponds to user's 'function_feed_forward')
        Propagates the input data through the 3 layers.
        Calculates weighted sums and applies the activation function.
        Returns the final probability score / classification guess.
        """
        # Save intermediate states for use in backpropagation
        self.X_input = X

        # Input -> Hidden 1
        self.Z1 = np.dot(self.X_input, self.W1) + self.b1
        self.A1 = self._sigmoid(self.Z1)

        # Hidden 1 -> Hidden 2
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self._sigmoid(self.Z2)

        # Hidden 2 -> Output
        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        self.output = self._sigmoid(self.Z3)

        return self.output

    def backpropagate(self, X, y, output_guess, learning_rate):
        """
        (Corresponds to user's 'function_train' core logic)
        Calculates the gradient of the loss function using the chain rule.
        Adjusts the weights and biases (synaptic strengths) via gradient descent.
        """
        m = X.shape[0]  # Number of samples in this batch

        # Calculate output layer error
        # Assuming simple Mean Squared Error for loss: L = 0.5 * (y - output_guess)^2
        error = y - output_guess

        # 1. Output Layer Gradients
        d_output = error * self._sigmoid_derivative(output_guess)
        dW3 = np.dot(self.A2.T, d_output) / m
        db3 = np.sum(d_output, axis=0, keepdims=True) / m

        # 2. Hidden Layer 2 Gradients
        error_h2 = np.dot(d_output, self.W3.T)
        d_h2 = error_h2 * self._sigmoid_derivative(self.A2)
        dW2 = np.dot(self.A1.T, d_h2) / m
        db2 = np.sum(d_h2, axis=0, keepdims=True) / m

        # 3. Hidden Layer 1 Gradients
        error_h1 = np.dot(d_h2, self.W2.T)
        d_h1 = error_h1 * self._sigmoid_derivative(self.A1)
        dW1 = np.dot(self.X_input.T, d_h1) / m
        db1 = np.sum(d_h1, axis=0, keepdims=True) / m

        # Gradient Descent Update (Adding because error was defined as y - output_guess)
        self.W3 += learning_rate * dW3
        self.b3 += learning_rate * db3
        self.W2 += learning_rate * dW2
        self.b2 += learning_rate * db2
        self.W1 += learning_rate * dW1
        self.b1 += learning_rate * db1

    def train(self, X_train, y_train, epochs, learning_rate):
        """
        Loops through the dataset for a given number of epochs.
        Calls feed_forward and backpropagate to train the network iteratively.
        Prints loss metrics to track diagnostic error reduction.
        """
        loss_history = []  # Create an empty list to store the history

        for epoch in range(epochs):
            # 1. Forward Pass
            output_guess = self.feed_forward(X_train)

            # 2. Calculate Loss (Mean Squared Error)
            loss = np.mean(np.square(y_train - output_guess))
            loss_history.append(loss)  # Save the loss for this epoch

            # 3. Backward Pass
            self.backpropagate(X_train, y_train, output_guess, learning_rate)

            # 4. Print diagnostic output
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch} | Loss: {loss:.4f}")

        return loss_history  # Hand the list back to main.py
