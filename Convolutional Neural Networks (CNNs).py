import numpy as np
# Convolution Layer
class ConvolutionLayer:
    def __init__(self, num_filters, filter_size):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.filters = np.random.randn(num_filters, filter_size, filter_size) / 9

    def iterate_regions(self, image):
        h, w = image.shape
        for i in range(h - self.filter_size + 1):
            for j in range(w - self.filter_size + 1):
                yield image[i:i+self.filter_size, j:j+self.filter_size], i, j

    def forward(self, input):
        self.last_input = input
        h, w = input.shape
        output = np.zeros((h - self.filter_size + 1, w - self.filter_size + 1, self.num_filters))
        for region, i, j in self.iterate_regions(input):
            output[i, j] = np.sum(region * self.filters, axis=(1, 2))
        return output

    def backward(self, d_L_d_out, learning_rate):
        d_L_d_filters = np.zeros(self.filters.shape)
        for region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                d_L_d_filters[f] += d_L_d_out[i, j, f] * region
        self.filters -= learning_rate * d_L_d_filters

# Max Pooling Layer
class MaxPoolingLayer:
    def __init__(self, pool_size):
        self.pool_size = pool_size

    def iterate_regions(self, image):
        h, w, num_filters = image.shape
        new_h = h // self.pool_size
        new_w = w // self.pool_size
        for i in range(new_h):
            for j in range(new_w):
                region = image[
                    i * self.pool_size:(i + 1) * self.pool_size,
                    j * self.pool_size:(j + 1) * self.pool_size
                ]
                yield region, i, j

    def forward(self, input):
        self.last_input = input
        h, w, num_filters = input.shape
        output = np.zeros((h // self.pool_size, w // self.pool_size, num_filters))
        for region, i, j in self.iterate_regions(input):
            output[i, j] = np.max(region, axis=(0, 1))
        return output

    def backward(self, d_L_d_out):
        d_L_d_input = np.zeros(self.last_input.shape)
        for region, i, j in self.iterate_regions(self.last_input):
            h, w, f = region.shape
            max_val = np.max(region, axis=(0, 1))
            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        if region[i2, j2, f2] == max_val[f2]:
                            d_L_d_input[
                                i * self.pool_size + i2,
                                j * self.pool_size + j2,
                                f2
                            ] = d_L_d_out[i, j, f2]
        return d_L_d_input

# Fully Connected Layer
class FullyConnectedLayer:
    def __init__(self, input_len, output_len):
        self.weights = np.random.randn(input_len, output_len) / input_len
        self.biases = np.zeros(output_len)

    def forward(self, input):
        self.last_input_shape = input.shape
        input = input.flatten()
        self.last_input = input
        self.last_output = np.dot(input, self.weights) + self.biases
        return self.last_output

    def backward(self, d_L_d_out, learning_rate):
        d_L_d_input = np.dot(d_L_d_out, self.weights.T)
        d_L_d_weights = np.dot(self.last_input[:, np.newaxis], d_L_d_out[np.newaxis, :])
        d_L_d_biases = d_L_d_out
        self.weights -= learning_rate * d_L_d_weights
        self.biases -= learning_rate * d_L_d_biases
        return d_L_d_input.reshape(self.last_input_shape)

# CNN class
class CNN:
    def __init__(self):
        self.conv = ConvolutionLayer(8, 3)
        self.pool = MaxPoolingLayer(2)
        self.fc = FullyConnectedLayer(8 * 3 * 3, 10)

    def forward(self, X):
        out = self.conv.forward(X)
        out = self.pool.forward(out)
        out = self.fc.forward(out)
        return out

    def backward(self, d_L_d_out, learning_rate):
        grad = self.fc.backward(d_L_d_out, learning_rate)
        grad = self.pool.backward(grad)
        self.conv.backward(grad, learning_rate)

    def train(self, X, Y, epochs, learning_rate):
        for epoch in range(epochs):
            loss = 0
            for i in range(len(X)):
                out = self.forward(X[i])
                exp_scores = np.exp(out - np.max(out))
                probs = exp_scores / np.sum(exp_scores)
                loss += -np.log(probs[Y[i].argmax()])
                d_L_d_out = probs
                d_L_d_out[Y[i].argmax()] -= 1
                self.backward(d_L_d_out, learning_rate)
            loss /= len(X)
            print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")

    def predict(self, X):
        predictions = []
        for i in range(len(X)):
            out = self.forward(X[i])
            predictions.append(np.argmax(out))
        return np.array(predictions)

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

# Load the Digits dataset (MNIST-like, smaller size)
digits = load_digits()
X = digits.images  # Images are 8x8 grayscale images
Y = digits.target.reshape(-1, 1)

# Normalize the data
X = X / 16.0

# One-hot encode the labels
encoder = OneHotEncoder(sparse_output=False)
Y_encoded = encoder.fit_transform(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_encoded, test_size=0.2, random_state=42)


cnn = CNN()
cnn.train(X_train, Y_train, epochs=3, learning_rate=0.01)

# Test the CNN
Y_pred = cnn.predict(X_test)
Y_test_classes = np.argmax(Y_test, axis=1)

# Calculate accuracy
accuracy = accuracy_score(Y_test_classes, Y_pred)
print(accuracy)
