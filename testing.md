test_preprocessing_normalization():
Passes a dummy array with values [0, 128, 255] to the loading function and asserts that the output is exactly [0.0, 0.501, 1.0]. Checks that the 2D array was successfully flattened to 1D.

test_feed_forward_dimensions(): 
Initializes the MultiLayerPerceptron, passes a single dummy image, and asserts that the output array shape perfectly matches the number of expected output classes.

test_backpropagation_learning(): 
Feeds the same dummy image to the network 50 times in a row, triggering the train function. It asserts that the loss/error calculation on the 50th run is strictly lower than the loss on the 1st run, proving the calculus and gradient descent are mathematically updating the weights in the correct direction. If the loss does not decrease, the weights are not changing/not changing in correct direction.
