import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
class KNearestNeighboursClassifier:

    """ 
    k is a hyperparameter that  defines how  many closest 
    datapoints are considered while making the prediction 
    here atleast 3.

    """
    
    def __init__(self,k=3) -> None:
        self.k=k
    
    """
    X_train and y_train are numpy arrays and store the data. 

    """
    
    def fit(self,X_train,y_train):
        self.X_train=X_train
        self.y_train=y_train

    """
    Takes the input in the form of the x_test and then returns 
    the predictions here x_test is a 2D array and then it takes 
    the 1D arrays one by one and returns the output.

    """
    def predict(self,X_test):
        predictions=[self._predict(x)for x in X_test]
        return np.array(predictions)


    def _predict(self,x):

        """
        Calculating the euclidean distances between x and 
        all the points in the training set.

        """
        distances=[self._eucledian_distance(x,x_train) for x_train in self.X_train]
        
        """
        argsort the indices in an ascending order that would
        sort the distances in ascending order.

        self.k slices the forst k indices which corresponds 
        to the k small distances. 

        Let’s say distances = [1.5, 0.3, 2.1, 0.7].
        np.argsort(distances) would return [1, 3, 0, 2] which 
        are the indices in sorted order
        
        self.k=2, k_indices = np.argsort(distances)[:2] would return [1,3]
        
        """
        k_indices=np.argsort(distances)[:self.k]

        """
        k_indices: contains the labels of the k nearest neighbours,
        as determined in the previous step.   

        self.y_train[i]: accesses the label/ target value of the 
        training data at index i.

        k_nearest_labels stores the labels of these k nearest neighbours.

        Example: If k_indices = [1, 3, 0], 
                    self.y_train=[0, 1, 1, 0] i.e. labels of the training data
                        
                        self.y_train[1] = 1
                        self.y_train[3] = 0
                        self.y_train[0] = 0

                 It extracts the labels of k nearest neighbours so that we can
                 perform the majority voting in case of classification

        """

        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # for a in k_nearest_labels:
        #     print(a)

        """
        Here we perform majority voting to determine the predicted class based 
        on the labels of the nearest neighbours.
        
        Example of counter:
            Counter([0, 1, 1]).most_common(1)
            Output: [(1, 2)]

        """

        most_common = Counter(k_nearest_labels).most_common(1)

        """
        most_common[0][0] gives the label 1, which is the predicted class
        
        """
        return most_common[0][0]

    
    def _eucledian_distance(self,x1,x2):
        return np.sqrt(np.sum((x1-x2)**2))
    
    


        