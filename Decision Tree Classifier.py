import numpy as np

class DecisionTreeClassifier:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.tree = None

        """
        self.max_depth: controls the maximum depth of the tree 
                        so that it does'nt grow indefinitely.
        
        self.tree: initially it is stored as None, later it 
                   stores the trained decision tree.

        """

    def fit(self,X,y):
        self.tree=self.__build_tree(X,y)

        """
        fit: used to train the model on the input data X and y.
             It calls the _build_tree to build the decision tree 
             recursively and the result is stored in self.tree.

        """   
    

    def predict(self,X):
        return np.array([self._predict_single(row) for row in X])

    """
    predict: This method takes a set of input features X and predicts 
             the labels for each row or instance. 
             It uses list comprehension to iterate over each row 
             (data sample) in X and calls _predict_single on each row 
             to get its prediction.
             These values are returned as a numpy array. 

    """

    def _build_tree(self, X, y, depth=0):
        n_samples,n_features=X.shape
        unique_labels=np.unique(y)

        if len(unique_labels)==1 or depth==self.max_depth or n_samples<self.min_samples_split:
            return {'label':self._most_common_label(y)}
    """
    build_tree: A recursive method that builds the decision tree.
    X.shape: gives the number of samples.
    
    
    """
