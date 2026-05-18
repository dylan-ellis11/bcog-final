# Multi-Layer Perceptron from Scratch: 

This project features a custom 3-layer Multi-Layer Perceptron (MLP) designed to classify chest X-rays from the PneumoniaMNIST dataset into X-rays of those with Pneumonia and X-rays of those without Pneumonia. Rather than relying on high-level machine learning frameworks like PyTorch, TensorFlow, or Scikit-learn, this neural network is implemented entirely from scratch using Python and NumPy. The primary goal of this architecture is to demonstrate the underlying mathematics of deep learning and the linear algebra required for feed-forward propagation and the calculus (chain rule) that allows for gradient descent and backpropagation. By classifying healthy vs. unhealthy lung scans, this project serves as a foundational AI system for medical diagnostic assistance. 

## How to Run and Test the Project:

Follow these steps to initialize the dataset, verify the mathematical architecture, and train the model.

1. Download the Data:
First, you need to pull the PneumoniaMNIST dataset into your local project directory. In the your terminal within the repository, run:

python data_draw.py

2. Verify Data and Architecture (Testing):
Before training, it is recommended to run the testing suite. This ensures that the data normalizes correctly, the matrix dimensions align across all three layers, and the backpropagation calculus successfully reduces loss. To do this, run: 

python testing_model.py.

Note: You can also optionally run python shape_find.py or python dataset_explorer.py to explore the underlying numpy arrays and class distributions.

3. Train the Model:
Once the data is downloaded and the architecture is verified, execute the main training loop. This will instantiate the model, train it over 2500 epochs, generate a matplotlib graph comparing Training vs. Validation Loss to monitor for overfitting, and output a final test accuracy score. To do this, run: 
python main.py
## File Structure and Explanations
### Core Engine: 
mlp_classifier.py is the mathematical core of the project. It contains the following:
1. load_and_preprocess_data: A function that standardizes raw .npz image data by flattening 2D 28x28 images into 1D arrays of size 784 and normalizing pixel values to a 0.0 - 1.0 scale for mathematical stability.
2. MultiLayerPerceptron: The primary class defining the 3-layer network. It initializes random weights and biases, handles non-linear transformations using ReLU and Sigmoid activation functions, executes the feed_forward pass to generate predictions, and utilizes the backpropagate function to adjust synaptic strengths using Binary Cross-Entropy loss. The train method manages the epoch looping and loss tracking.

Execution Scripts:
1. main.py: The primary execution script. It initializes an MLP with a specific node architecture (784 Input $\rightarrow$ 128 Hidden $\rightarrow$ 64 Hidden $\rightarrow$ 1 Output). It handles the full pipeline: loading data, training for 2500 epochs, plotting the training/validation loss curves side-by-side, and printing the final exam accuracy.
2. main_alternative.py: An alternative execution script used for hyperparameter testing. It functions identically to main.py but tests a different number of nodes for its hidden layer architecture (784 Input $\rightarrow$ 256 Hidden $\rightarrow$ 128 Hidden $\rightarrow$ 1 Output) to observe differences in learning capacity and overfitting.

Data Management Utilities:
1. data_draw.py: A utility script that utilizes the medmnist library to securely download the Pneumonia dataset and copy the resulting pneumoniamnist.npz file directly into the working directory.
2. dataset_explorer.py: A simple diagnostic tool that loads the .npz file to calculate and print the class distribution (healthy vs. pneumonia) within the training data.
3. shape_find.py: An inspection script that iterates through the .npz file keys to print the array names, shapes, data types, and min/max values. Useful for confirming data structures before pushing them through the network.

Testing Suite:

To test that the model is initializing correctly and passes the following tests run the following:

python testing_model.py

testing_model.py is the testing file to validate the network's integrity before full-scale training. It contains the following functions:

1. test_preprocessing_normalization: Passes a dummy array to verify that 3D structures are properly flattened to 2D and normalized strictly between 0 and 1.
2. test_feed_forward_dimensions: Pushes a dummy image through the network to assert that the output matrix shape perfectly matches the expected 1x1 probability score.
3. test_backpropagation_learning: Feeds a dummy image through the training loop for 50 epochs. It strictly asserts that the 50th epoch's loss is lower than the 1st epoch's loss, definitively proving that the calculus and gradient descent are mathematically updating the weights in the correct direction.

## Optional Tests to Run and Changes to Make:
If interested, you can make alterations to see if your final test accuracy can improve. If you want to alter alter number of epochs, change epochs=2500 to epochs=n. Keep in mind it will result in longer compute time. If you want to try new architectures, you can change hidden_nodes_1=128, hidden_nodes_2=64 to hidden_nodes=n for each variable and see if the test accuracy improves. This could also result in longer compute time. Finally, changing the learning_rate variable within main.py or main_alternative.py can create possible improvements in accuracy.