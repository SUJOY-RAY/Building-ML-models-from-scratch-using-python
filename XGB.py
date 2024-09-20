import numpy as np
from sklearn.tree import DecisionTreeRegressor

class XGBClassifierFromScratch:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.gamma = []  # weights for each tree
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def log_loss(self, y_true, y_pred):
        # Logistic loss for binary classification
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def gradient(self, y_true, y_pred):
        # Gradient of the logistic loss function
        return y_pred - y_true

    def fit(self, X, y):
        # Initialize model with constant prediction (base score)
        y_pred = np.full(y.shape, np.mean(y))  # Initial predictions are the mean of labels

        for i in range(self.n_estimators):
            # Compute gradient (residuals)
            residuals = self.gradient(y, y_pred)

            # Train a decision tree on the residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X, residuals)
            
            # Make predictions with the tree
            tree_pred = tree.predict(X)
            
            # Compute the weight (gamma) of the tree
            gamma = self.learning_rate
            self.gamma.append(gamma)
            
            # Update predictions with the new tree
            y_pred += gamma * tree_pred
            
            # Save the tree
            self.trees.append(tree)

    def predict_proba(self, X):
        # Start with initial prediction (mean value of labels in case of binary classification)
        y_pred = np.zeros(X.shape[0])

        # Sum predictions from all trees
        for tree, gamma in zip(self.trees, self.gamma):
            y_pred += gamma * tree.predict(X)
        
        return self.sigmoid(y_pred)  # Use sigmoid to get probabilities for classification

    def predict(self, X):
        # Convert probabilities to class predictions (binary classification: 0 or 1)
        return (self.predict_proba(X) > 0.5).astype(int)


