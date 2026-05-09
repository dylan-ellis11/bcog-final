# bcog-final-project
My final project will be a 3 layer multi-layer perceptron built using Python and NumPy rather than using other libraries. This will be harder than a single layer perceptron as it will require calculus rather than simple arithmetic for one layer. This is going to use MedMNIST datasets on health images to determine biological problems within between healthy and unhealthy blood cells. This will be useful for building an AI system to help identify the condition/health of people who undergo blood tests.  

Requirements for data are either numpy files in .npz format or usage of .png or .jpg files in low pixel formats, like 28x28 as MedMNIST offers. The pixel values will float values between 0 and 1. Then, to correctly train the MLP, we will have training_images, training_labels, testing_images, and testing_labels. These labels for the data will help train and test the MLP.

load_and_preprocess_data(filepath): This function reads the raw medical image data from the specified file path. It standardizes the data by converting images to grayscale, resizing them to uniform dimensions, and flattening them into 1D arrays. Finally, it normalizes the pixel values to a $0.0 - 1.0$ scale to ensure mathematical stability during training.  

__init__(self, input_nodes, hidden_nodes_1, hidden_nodes_2, output_nodes): This constructor initializes the structural architecture of the 3-layer perceptron. It sets the size of the input, hidden, and output layers based on the parameters provided. It generates the initial weight matrices and bias vectors with small random numbers to prepare the network for learning.  

_sigmoid(self, x): Function acts as the activation function for the network's layers. It mathematically squishes any input value into a range between 0 and 1. This non-linear transformation is what allows the multi-layer network to learn complex patterns other than simple linear regression.

_sigmoid_derivative(self, x): This function calculates the derivative (slope) of the sigmoid curve at any given point. It is a necessary calculus component used for the backpropagation to determine the direction and magnitude of the error gradient.  

feed_forward(self, X): This function pushes the normalized image data forward through the network's layers. It calculates the dot product of the inputs and weights, adds the biases, and passes the results through the sigmoid activation function. The final output is an array of probability scores representing the model's diagnostic guess.  

backpropagate(self, X, y, output_guess, learning_rate): This function calculates the network's diagnostic error by comparing its prediction to the actual medical label. Using the chain rule of calculus, it determines how much each weight and bias contributed to that error. It then updates those synaptic strengths via gradient descent to improve future accuracy.  

train(self, X_train, y_train, epochs, learning_rate)This function manages the overall learning loop by repeatedly feeding the training dataset through the network over a specified number of epochs. In each iteration, it calls both feed_forward to get a prediction and backpropagate to update the weights. It also tracks and prints the decreasing loss metric so users can monitor the learning progress.