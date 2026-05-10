import numpy as np
import os
from mlp_classifier import MultiLayerPerceptron, load_and_preprocess_data


def test_preprocessing_normalization():
    print("Running test_preprocessing_normalization...")

    # 1. Create a dummy dataset with the exact values from your testing.md
    dummy_train_images = np.array([[[0, 128, 255]]], dtype=np.uint8)
    dummy_train_labels = np.array([1])

    # ADDED: Dummy validation data to satisfy load_and_preprocess_data
    dummy_val_images = np.array([[[0, 128, 255]]], dtype=np.uint8)
    dummy_val_labels = np.array([1])

    dummy_test_images = np.array([[[0, 128, 255]]], dtype=np.uint8)
    dummy_test_labels = np.array([1])

    # 2. Save it to a temporary .npz file to mimic MedMNIST
    temp_filepath = "temp_dummy_data.npz"
    np.savez(
        temp_filepath,
        train_images=dummy_train_images,
        train_labels=dummy_train_labels,
        val_images=dummy_val_images,  # <--- ADDED
        val_labels=dummy_val_labels,  # <--- ADDED
        test_images=dummy_test_images,
        test_labels=dummy_test_labels,
    )

    try:
        # 3. Pass the filepath to your function
        # NOTE: If your function returns validation data too, you will need to
        # unpack 6 variables here instead of 4 (e.g., X_train, y_train, X_val, y_val, X_test, y_test)
        X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data(
            temp_filepath
        )

        # 4. Check if it flattened from 3D to 2D (1 image, 3 pixels)
        assert X_train.shape == (1, 3), f"Expected shape (1, 3), got {X_train.shape}"

        # 5. Check if it normalized correctly
        expected_values = np.array([[0.0, 128 / 255.0, 1.0]])
        # np.allclose handles tiny floating point differences (like 0.50196 vs 0.501)
        assert np.allclose(
            X_train, expected_values, atol=1e-3
        ), f"Normalization failed. Got {X_train}"

        print(" -> SUCCESS: Data successfully flattened and normalized!\n")
    finally:
        # Clean up the temporary file so it doesn't clutter your folder
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)


def test_feed_forward_dimensions():
    print("Running test_feed_forward_dimensions...")

    # Initialize network (784 input pixels, 2 hidden layers, 1 output class for healthy/unhealthy)
    model = MultiLayerPerceptron(
        input_nodes=784, hidden_nodes_1=64, hidden_nodes_2=32, output_nodes=1
    )

    # Create a single dummy image (1 sample, 784 pixels)
    dummy_image = np.random.rand(1, 784)

    # Push it through the network
    output = model.feed_forward(dummy_image)

    # Assert the output is a single probability score (1 sample, 1 output)
    assert output.shape == (1, 1), f"Expected output shape (1, 1), got {output.shape}"
    assert (
        0.0 <= output[0][0] <= 1.0
    ), "Output must be squished between 0 and 1 by the sigmoid function"

    print(" -> SUCCESS: Matrix dimensions align perfectly through all 3 layers!\n")


def test_backpropagation_learning():
    print("Running test_backpropagation_learning...")

    # Initialize network
    model = MultiLayerPerceptron(
        input_nodes=784, hidden_nodes_1=64, hidden_nodes_2=32, output_nodes=1
    )

    # Create a single dummy image and a target label (e.g., label = 1 for unhealthy)
    dummy_image = np.random.rand(1, 784)
    target_label = np.array([[1.0]])

    # ADDED: Create dummy validation data to satisfy the train method's requirements
    dummy_val_image = np.random.rand(1, 784)
    val_target_label = np.array([[0.0]])

    # 1. Get the baseline loss BEFORE any training
    initial_guess = model.feed_forward(dummy_image)
    initial_loss = np.mean(np.square(target_label - initial_guess))

    # 2. Train the network on this single image for 50 epochs
    # We use a relatively high learning rate here just to force a noticeable change quickly
    model.train(
        X_train=dummy_image,
        y_train=target_label,
        X_val=dummy_val_image,  # <--- ADDED
        y_val=val_target_label,  # <--- ADDED
        epochs=50,
        learning_rate=0.1,
    )

    # 3. Get the new loss AFTER training
    final_guess = model.feed_forward(dummy_image)
    final_loss = np.mean(np.square(target_label - final_guess))

    # 4. The absolute proof that calculus is working
    assert (
        final_loss < initial_loss
    ), f"Loss did not decrease! Initial: {initial_loss:.4f}, Final: {final_loss:.4f}"

    print(
        f" -> SUCCESS: The chain rule works. Loss decreased from {initial_loss:.4f} to {final_loss:.4f}!\n"
    )


if __name__ == "__main__":
    print("=== Starting MLP Architecture Tests ===\n")
    test_preprocessing_normalization()
    test_feed_forward_dimensions()
    test_backpropagation_learning()
    print("=== All tests passed! You are ready for MedMNIST data. ===")
