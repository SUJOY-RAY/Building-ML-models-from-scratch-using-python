import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

model=Sequential([
    Dense(4,input_dim=2,activation='relu'),
    Dense(4,activation='relu'),
    Dense(1,activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.01),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X,y,epochs=1000,verbose=0)
loss,accuracy=model.evaluate(X,y,verbose=0)
print(f'Loss: {loss}, Accuracy: {accuracy}')

predictions=(model.predict(X)>0.5).astype(int)
print("Predictions: ",predictions.flatten())
