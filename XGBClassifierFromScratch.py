import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor


class XGBClassifier:
    def __init__(self, n_estimators = 100,learning_rate = 0.1,max_depth = 3,reg_lambda = 1.0):
        self.n_estimators = n_estimators              # Number of boosting rounds (trees)
        self.learning_rate = learning_rate            # Learning rate (shrinkage)
        self.max_depth = max_depth                    # Maximum depth of trees
        self.reg_lambda = reg_lambda                  # Regularization parameter (L2)
        self.trees = []                               # List to store each decision tree
        self.base_pred = None                         # Base prediction (initial prediction)

    
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    

    def _log_loss_gradient_hessian(self,y_true,y_pred):
        pred_prob = self.sigmoid(y_pred)
        grad = pred_prob-y_true
        hess = pred_prob*(1-pred_prob)
        return grad, hess
    

    def fit(self, X, y):
        n_samples = X.shape[0]

        pos_ratio = np.sum(y)/len(y)
        self.base_pred = np.log(pos_ratio/(1-pos_ratio))
        y_pred = np.full(n_samples,self.base_pred)    # y_pred = np.full(5, 0.5)
                                                      # y_pred: array([0.5, 0.5, 
                                                      #                0.5, 0.5, 0.5])
        
        for i in range(self.n_estimators):
            grad, hess = self._log_loss_gradient_hessian(y, y_pred)
            
            tree = DecisionTreeRegressor(max_depth = self.max_depth)
            tree.fit(X,-grad/(hess + self.reg_lambda))
            
            y_pred += self.learning_rate*tree.predict(X)
            
            self.trees.append(tree)
        
    
    def predict(self, X):
        y_pred = np.full(X.shape[0],self.base_pred)
        
        for tree in self.trees:
            y_pred += self.learning_rate*tree.predict(X)
        
        return np.round(self.sigmoid(y_pred))

    
    def predict_proba(self, X):
        y_pred = np.full(X.shape[0], self.base_pred)
        for tree in self.trees:
            y_pred += self.learning_rate*tree.predict(X)
        
        return self.sigmoid(y_pred)