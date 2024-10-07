import numpy as np
from collections import Counter


# Helper functions

""" This function computes the entropy, a measure of impurity or uncertainty in a set of labels """
def entropy(y):
    hist=np.bincount(y)    
    """ This counts occurences if each class in the label array y """
    
    ps=hist/len(y)
    """ This computes the probabilities of each class """
    
    return -np.sum([p*np.log2(p) for p in ps if ps>0])
    """ This is the entropy formula """


""" This function splits the dataset into two parts based on threshold value of a specific feature """
def split_dataset(X,y,feature_index, threshold):
    left_mask=X[:,feature_index]<=threshold
    right_mask=X[:,feature_index]>threshold
    """ left_mask and right_mask are boolean arrays indicating which samples go to the left or right split """

    return X[left_mask],X[right_mask],y[left_mask],y[right_mask]
    """ This function returns the split datsets: features X and labels Y """ 


def information_gain(y,y_left,y_right):
    weight_left=len(y_left)/len(y)
    weight_right=len(y_right)/len(y)
    """ weight_left, weight_right are proportions of samples in the left and right splits """

    return entropy(y)-(weight_left*entropy(y_left)+weight_right*entropy(y_right))
    """ This formula substracts the weighted average of entropies in the left and right splits from the original entropy of the data """


# Decision Tree Class
class DecisionTree:
    def __init__(self,max_depth=None, min_samples_split=2):
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.tree=None

    """
    max_depth: To control overfitting
    min_samples_split: The minimum number of samples required to split a node
    self.tree: Stores the constructed tree as a dictionary. 
    
    """

    def _best_split(self,X,y):
        best_gain=-1
        best_split=None
        n_samples, n_features=X.shape

        """
        best_gain: Stores the best information gain found
        best_split: Stores the best split configuration.
        n_samples, n_features: These are the number of sampels  and features in X

        """
        
        for feature_index in range(n_features):
            thresholds=np.unique(X[:feature_index])
            for threshold in thresholds:
                X_left, X_right, y_left, y_right =split_dataset(X, y, feature_index,threshold)
                if len(y_left)>0 and len(y_right)>0:
                    gain=information_gain(y,y_left,y_right)
                    """
                    If both sides of the split have samples, calculate the information gain.
                    """

                    if gain>best_gain:
                        best_gain=gain
                        best_split={
                            'feature_index':feature_index,
                            'threshold':threshold,
                            'X_left':X_left,
                            'X_right':X_right,
                            'y_left':y_left,
                            'y_right':y_right
                        }
                    """
                    If the current split gives a better gain than previous splits, update best_gain and store the split information in best_split.
                    """
        return best_split
    
    def 
                    



