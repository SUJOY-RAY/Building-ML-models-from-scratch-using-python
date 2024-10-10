# import KNN
#
# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
#
# X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# knn = KNN.KNearestNeighboursClassifier(4)
# knn.fit(X_train, y_train)
#
# y_pred = knn.predict(X_test)
#
# print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# import XGB
# import numpy as np
# # Sample data for testing the implementation
# X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
# y_train = np.array([0, 0, 1, 1, 1])
# X_test = np.array([[1, 2], [3, 4], [5, 6]])
#
# # Instantiate the model
# xgb = XGB.XGBClassifierFromScratch(n_estimators=10, learning_rate=0.1, max_depth=3)
#
# # Fit the model on training data
# xgb.fit(X_train, y_train)
#
# # Predict on test data
# y_pred = xgb.predict(X_test)
# print(y_pred)
