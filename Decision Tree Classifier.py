import numpy as np

class DecisionTreeClassifier:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.tree = None
    
    def fit(self, X, y):
        self.tree = self._build_tree(X, y)
    
    def predict(self, X):
        return np.array([self._predict_single(row) for row in X])
    
    def _build_tree(self, X, y, depth=0):
        # Check if all labels are the same or max depth is reached
        if len(np.unique(y)) == 1 or depth == self.max_depth:
            return {'label': np.bincount(y).argmax()}
        
        feature, threshold = self._best_split(X, y)
        left_indices = X[:, feature] <= threshold
        right_indices = ~left_indices
        
        left_tree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_tree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return {
            'feature': feature,
            'threshold': threshold,
            'left': left_tree,
            'right': right_tree
        }
    
    def _best_split(self, X, y):
        best_gini = 1
        best_feature = None
        best_threshold = None

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gini = self._calculate_gini(X[:, feature], y, threshold)
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold
    
    def _calculate_gini(self, feature_values, y, threshold):
        left_indices = feature_values <= threshold
        right_indices = ~left_indices

        # Handle empty splits
        if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
            return 1

        left_gini = self._gini(y[left_indices])
        right_gini = self._gini(y[right_indices])

        # Weighted average of Gini impurity
        gini = (len(y[left_indices]) * left_gini + len(y[right_indices]) * right_gini) / len(y)
        return gini
    
    def _gini(self, y):
        proportions = np.bincount(y) / len(y)
        return 1 - np.sum(proportions**2)
    
    def _predict_single(self, row):
        node = self.tree
        while 'label' not in node:
            if row[node['feature']] <= node['threshold']:
                node = node['left']
            else:
                node = node['right']
        return node['label']

# Example usage:
X = np.array([[2, 3], [1, 5], [4, 6], [7, 8], [1, 2]])
y = np.array([0, 0, 1, 1, 0])
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X, y)
predictions = clf.predict(X)
print(predictions)