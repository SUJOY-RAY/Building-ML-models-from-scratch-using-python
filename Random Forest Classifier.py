from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
import numpy as np


class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, Y):
        self.root = self._build_tree(X, Y)

    def _build_tree(self, X, Y, depth=0):
        num_samples, num_features = X.shape

        # If no samples or max depth reached, return majority class
        if num_samples == 0 or (self.max_depth is not None and depth >= self.max_depth):
            return Counter(Y).most_common(1)[0][0]

        best_feature, best_threshold = self._find_best_split(X, Y)

        # If no split is possible, return majority class
        if best_feature is None:
            return Counter(Y).most_common(1)[0][0]

        left_indices = X[:, best_feature] <= best_threshold
        right_indices = ~left_indices

        # If any split is empty, return majority class
        if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
            return Counter(Y).most_common(1)[0][0]

        left_child = self._build_tree(X[left_indices], Y[left_indices], depth + 1)
        right_child = self._build_tree(X[right_indices], Y[right_indices], depth + 1)

        return {"feature": best_feature, "threshold": best_threshold, "left": left_child, "right": right_child}

    def _find_best_split(self, X, Y):
        best_feature, best_threshold, best_gini = None, None, float("inf")
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] <= threshold
                right_indices = ~left_indices
                gini = self._gini(Y[left_indices], Y[right_indices])
                if gini < best_gini:
                    best_gini, best_feature, best_threshold = gini, feature, threshold
        return best_feature, best_threshold

    def _gini(self, left, right):
        def calculate_gini(y):
            if len(y) == 0:
                return 0
            proportions = np.bincount(y) / len(y)
            return 1 - np.sum(proportions ** 2)

        n_left, n_right = len(left), len(right)
        total = n_left + n_right
        return (n_left / total) * calculate_gini(left) + (n_right / total) * calculate_gini(right)

    def predict(self, X):
        return np.array([self._predict_single(x, self.root) for x in X])

    def _predict_single(self, x, tree):
        if isinstance(tree, dict):
            feature, threshold = tree["feature"], tree["threshold"]
            if x[feature] <= threshold:
                return self._predict_single(x, tree["left"])
            else:
                return self._predict_single(x, tree["right"])
        else:
            return tree


class RandomForestClassifier:
    def __init__(self, num_trees=10, max_depth=None, max_features="sqrt"):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []

    def fit(self, X, Y):
        num_samples, num_features = X.shape
        for _ in range(self.num_trees):
            bootstrap_indices = np.random.choice(num_samples, num_samples, replace=True)
            feature_indices = self._get_feature_indices(num_features)
            X_bootstrap, Y_bootstrap = X[bootstrap_indices][:, feature_indices], Y[bootstrap_indices]

            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_bootstrap, Y_bootstrap)
            self.trees.append((tree, feature_indices))

    def _get_feature_indices(self, num_features):
        if self.max_features == "sqrt":
            return np.random.choice(num_features, int(np.sqrt(num_features)), replace=False)
        elif isinstance(self.max_features, int):
            return np.random.choice(num_features, self.max_features, replace=False)
        else:
            return np.arange(num_features)

    def predict(self, X):
        predictions = []
        for tree, feature_indices in self.trees:
            predictions.append(tree.predict(X[:, feature_indices]))
        predictions = np.array(predictions).T
        return np.array([Counter(row).most_common(1)[0][0] for row in predictions])


# Test the RandomForestClassifier on the Iris dataset
iris = load_iris()
X, Y = iris.data, iris.target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train the Random Forest Classifier
rf = RandomForestClassifier(num_trees=10, max_depth=5, max_features="sqrt")
rf.fit(X_train, Y_train)

# Predict and evaluate
Y_pred = rf.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
print(accuracy)

