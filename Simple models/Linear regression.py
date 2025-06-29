import numpy as np
import matplotlib.pyplot as plt


X = np.array([x for x in range(1000)])
y = np.array([x+2 for x in range(1000)])


x_mean = np.mean(X)
y_mean = np.mean(y)

numerator = np.sum((X - x_mean) * (y - y_mean))
denominator = np.sum((X - x_mean)**2)
slope = numerator / denominator

"""

m = ∑(xi - x_bar)(yi - y_bar) 
          ∑(xi-x_bar)^2

"""

intercept = y_mean - slope * x_mean

"""
y = m * x + c
c = y - m * x

"""


def predict(x):
    return slope * X + intercept


y_pred = predict(X)

plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, y_pred, color='red', label='Predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()
plt.show()


print(f"Slope (m): {slope}")
print(f"Intercept (b): {intercept}")


""" Value for x = 400 """
val = 400
result = slope * 400 + intercept
print(result)
