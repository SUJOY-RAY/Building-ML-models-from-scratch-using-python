import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=10000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None 
        self.bias = None 
    
    def fit(self, X: np.ndarray , y: np.ndarray) -> None:
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            y_predicted = np.dot(X, self.weights) + self.bias

            dw = np.dot(X.T, (y_predicted - y))/n_samples
            db = np.sum(y_predicted - y)/n_samples
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X : np.ndarray) -> np.ndarray:
        return np.dot(X, self.weights) + self.bias

if __name__ == "__main__":
    # Sample data
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])  # y = 2x

    model = LinearRegression(learning_rate=0.01, epochs=1000)
    model.fit(X, y)
    predictions = model.predict(X)

    print("Weights:", model.weights)
    print("Bias:", model.bias)
    print("Predictions:", predictions)
