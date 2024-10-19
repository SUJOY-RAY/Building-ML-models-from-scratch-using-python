# Import libraries
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


# Load example dataset
iris = load_iris()
X, y = iris.data, iris.target

# Create and train the decision tree classifier
clf = DecisionTreeClassifier()
clf = clf.fit(X, y)

# Plot the decision tree
plt.figure(figsize=(10, 8))  # Set figure size
tree.plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names, rounded=True)
plt.show()






import xgboost as xgb
import matplotlib.pyplot as plt



iris = load_iris()
X, y = iris.data, iris.target

# Create and train the decision tree classifier

clf = xgb.train(X,y)

xgb.plot_tree(clf)
plt.show()
