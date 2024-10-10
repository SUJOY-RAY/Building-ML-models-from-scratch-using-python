import numpy as np

class DecisionTreeClassifier:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split=min_samples_split
        self.tree = None

        """
        self.max_depth: controls the maximum depth of the tree 
                        so that it does'nt grow indefinitely.

        self.min_samples_split: The minimum number of samples 
                                required to split a node

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
        np.unique(y): returns the unique class labels in y(target variable)
        # the stopping conditions are checked :
            if all samples have same label (len(unique_labels))==1
            if maximum depth has been reached (depth==self.max_depth)
            if the number if samples in the current node is smaller than min_samples_split       
                # if any of these conditions are met, the node becomes a leaf, and we return 
                the most common label using _most_common_label.

        """

        best_feature,best_threshold=self._best_split(X,y)
        if best_feature is None:
            return {'label':self._most_common_label(y)}
        
        """
        if no stopping conditions are met, we find the best feature and 
        threshold to split the data using _best_split. 
        If  no split is found, (i.e. best_feature is None),the
        node becomes a leaf and returns the most common label.  

        """

        left_indices=X[:,best_feature]<=best_threshold
        right_indices=~left_indices
        
        """
        left_indices: This selects the rows where the best feature is 
                      less than or equal to the threshold.
        right_indices: This selects the remaining rows(those not in left_indices).

        """

        left_subtree=self._build_tree(X[left_indices],y[left_indices],depth+1)
        right_subtree=self._build_tree(X[right_indices],y[right_indices],depth+1)

        """
        Recursively build the left and right subtrees using the newly created subsets
        of data. The depth is incremented by 1 for each recursive call.

        """

        return{
            'feature':best_feature,
            'threshold':best_threshold,
            'left':left_subtree,
            'right':right_subtree
        }

        """
        feature: The feature used to split the data.
        threshold: The threshold value that splits the data.
        left: The left subtree.
        right: The right subtree.

        """
    def _best_split(self,X,y):
        n_samples,n_features=X.shape
        best_gini=1.0
        best_feature=None
        best_threshold=None
        for feature in range(n_features):
            thresholds=np.unique(X[:,feature])
            for threshold in thresholds:
                gini=self._calculate_gini(X[:,feature],y,threshold)
                if gini< best_gini:
                    best_gini=gini
                    best_feature=feature
                    best_threshold=threshold
        return best_feature, best_threshold





