import numpy as np

class Perceptron:
    def __init__(self,input_dim,learning_rate=0.01,epochs=1000) -> None:
        self.learning_rate=learning_rate
        self.epochs=epochs
        self.weights=np.zeros(input_dim+1)

    def activation(self,z):
        return np.where(z>0,1,0)
    
    def fit(self,X,y):
        X=np.c_[np.ones(X.shape[0]),X]
        for _ in range(self.epochs):
            for i in range(X.shape[0]):
                z=np.dot(X[i],self.weights)
                prediction=self.activation(z)
                error=y[i]-prediction
                self.weights+=self.learning_rate*error*X[i]
    
    def predict(self,X):
        X=np.c_[np.ones(X.shape[0]),X]
        z=np.dot(X,self.weights)
        return self.activation(z)
    
if __name__=="__main__":
    np.random.seed(0)
    X_class1 = np.random.randn(50, 2) + np.array([1, 1])
    X_class2 = np.random.randn(50, 2) + np.array([-1, -1])
    X = np.vstack((X_class1, X_class2))
    y = np.array([0] * 50 + [1] * 50)  # Labels: 0 for class1, 1 for class2


    perceptron=Perceptron(input_dim=2,learning_rate=0.1,epochs=10)
    perceptron.fit(X,y)

    predictions=perceptron.predict(X)
    print("Predictions: ",predictions)

