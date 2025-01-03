import numpy as np

class FeedforwardNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the feedforward neural network with one hidden layer.
        :param input_size: Number of input features.
        :param hidden_size: Number of neurons in the hidden layer.
        :param output_size: Number of output neurons (classes).
        """
        # Initialize weights and biases
        self.weights_input_hidden = np.random.rand(input_size, hidden_size) * 0.1
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.rand(hidden_size, output_size) * 0.1
        self.bias_output = np.zeros((1, output_size))

    def sigmoid(self, x):
        """
        Sigmoid activation function.
        """
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """
        Derivative of the sigmoid function.
        """
        return x * (1 - x)

    def forward(self, X):
        """
        Perform the forward pass.
        :param X: Input data.
        """
        # Hidden layer computations
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)

        # Output layer computations
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = self.sigmoid(self.final_input)
        return self.final_output

    def backward(self, X, Y, output, learning_rate):
        """
        Perform the backward pass and update weights and biases.
        :param X: Input data.
        :param Y: True labels.
        :param output: Predicted output from the forward pass.
        :param learning_rate: Learning rate for gradient descent.
        """
        # Calculate the error
        error = Y - output

        # Output layer gradients
        output_gradient = error * self.sigmoid_derivative(output)
        weights_hidden_output_gradient = np.dot(self.hidden_output.T, output_gradient)
        bias_output_gradient = np.sum(output_gradient, axis=0, keepdims=True)

        # Hidden layer gradients
        hidden_error = np.dot(output_gradient, self.weights_hidden_output.T)
        hidden_gradient = hidden_error * self.sigmoid_derivative(self.hidden_output)
        weights_input_hidden_gradient = np.dot(X.T, hidden_gradient)
        bias_hidden_gradient = np.sum(hidden_gradient, axis=0, keepdims=True)

        # Update weights and biases
        self.weights_hidden_output += learning_rate * weights_hidden_output_gradient
        self.bias_output += learning_rate * bias_output_gradient
        self.weights_input_hidden += learning_rate * weights_input_hidden_gradient
        self.bias_hidden += learning_rate * bias_hidden_gradient

    def train(self, X, Y, epochs, learning_rate):
        """
        Train the neural network.
        :param X: Training data.
        :param Y: Training labels.
        :param epochs: Number of epochs.
        :param learning_rate: Learning rate for gradient descent.
        """
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)

            # Backward pass
            self.backward(X, Y, output, learning_rate)

            # Print loss every 100 epochs
            if epoch % 100 == 0:
                loss = np.mean((Y - output) ** 2)
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        """
        Predict the output for given input data.
        :param X: Input data.
        """
        output = self.forward(X)
        return np.argmax(output, axis=1)

# Example: Training on a simple XOR problem
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y_train = np.array([[0], [1], [1], [0]])

# Create and train the feedforward neural network
fnn = FeedforwardNeuralNetwork(input_size=2, hidden_size=4, output_size=1)
fnn.train(X_train, Y_train, epochs=1000, learning_rate=0.1)

# Test the neural network
predictions = fnn.predict(X_train)
print("Predictions:", predictions)

