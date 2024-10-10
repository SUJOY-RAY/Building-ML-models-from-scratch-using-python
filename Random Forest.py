import numpy as np
from collections import Counter

# Helper functions

""" This function computes the entropy, a measure of impurity or uncertainty in a set of labels """


def entropy(y):
    hist = np.bincount(y)
    """ This counts occurences if each class in the label array y """

    ps = hist / len(y)
    """ This computes the probabilities of each class """

    return -np.sum([p * np.log2(p) for p in ps if ps > 0])
    """ This is the entropy formula """


""" This function splits the dataset into two parts based on threshold value of a specific feature """


def split_dataset(X, y, feature_index, threshold):
    left_mask = X[:, feature_index] <= threshold
    right_mask = X[:, feature_index] > threshold
    """ left_mask and right_mask are boolean arrays indicating which samples go to the left or right split """

    return X[left_mask], X[right_mask], y[left_mask], y[right_mask]
    """ This function returns the split datsets: features X and labels Y """


def information_gain(y, y_left, y_right):
    weight_left = len(y_left) / len(y)
    weight_right = len(y_right) / len(y)
    """ weight_left, weight_right are proportions of samples in the left and right splits """

    return entropy(y) - (weight_left * entropy(y_left) + weight_right * entropy(y_right))
    """ This formula substracts the weighted average of entropies in the left and right splits from the original entropy of the data """


# Decision Tree Class
class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    """
    max_depth: To control overfitting
    min_samples_split: The minimum number of samples required to split a node
    self.tree: Stores the constructed tree as a dictionary. 
    
    """

    def _best_split(self, X, y):
        best_gain = -1
        best_split = None
        n_samples, n_features = X.shape

        """
        best_gain: Stores the best information gain found
        best_split: Stores the best split configuration.
        n_samples, n_features: These are the number of sampels  and features in X

        """

        for feature_index in range(n_features):
            thresholds = np.unique(X[:feature_index])
            for threshold in thresholds:
                X_left, X_right, y_left, y_right = split_dataset(X, y, feature_index, threshold)
                if len(y_left) > 0 and len(y_right) > 0:
                    gain = information_gain(y, y_left, y_right)
                    """
                    If both sides of the split have samples, calculate the information gain.
                    """

                    if gain > best_gain:
                        best_gain = gain
                        best_split = {
                            'feature_index': feature_index,
                            'threshold': threshold,
                            'X_left': X_left,
                            'X_right': X_right,
                            'y_left': y_left,
                            'y_right': y_right
                        }
                    """
                    If the current split gives a better gain than previous splits, update best_gain and store the split information in best_split.
                    """
        return best_split

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        if n_samples >= self.min_samples_split and depth != self.max_depth:
            """
                It checks if the maximum depth of the tree has been reached or if there are enough samples.
            
            """
            split = self._best_split(X, y)

            if split:
                left_subtree = self._build_tree(split['X_left'], split['y_left'], depth + 1)
                right_subtree = self._build_tree(split['X_right'], split['y_right'], depth + 1)

                """
                if the best split is found then the function recursively builds left and right subtrees.

                """

                return {
                    'feature_index': split['feature_index'],
                    'threshold': split['threshold'],
                    'left': left_subtree,
                    'right': right_subtree
                }
            
            
            """
            If no split is found then it just returns the most common label in the current data (leaf node)

            """

            return Counter(y).most_common(1)[0][0]

    def fit(self,X,y):
        self.tree=self._build_tree(X,y)
    
    """
        Trains the decision tree by calling _build_tree to create the tree structure and stores it in self.tree. 

    """

    def _predict(self,x,tree):
        if isinstance(tree,dict):
            feature_index=tree['feature_index']
            threshold=tree['threshold']
            if x[feature_index]<=threshold:
                return self._predict(x,tree['left'])
            else:
                return self._predict(x,tree['right'])
        return tree
    """
        Predicts the class of a single sample x by traversing the tree from root to leaf.
            
            @if the current node is a decision node then it decides on the basis of the threshold value whether to go left or right.

            @If the leaf node is reached then it returns the predicted class.
    
    """


    def predict(self,X):
        return [self._predict(x,self.tree) for x in X]
    
    """
        Predicts for multiple samples by calling _predcit on each sample.

    """



"""

Implememting the Random Forest Classifier

"""




class RandomForest:
    def __init__(self,n_trees=10, max_depth=None, min_samples_split=2,max_features=None):
        self.n_trees=n_trees
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.max_features=max_features
        self.trees=[]
        
        