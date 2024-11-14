from dense_layer import Dense
from hyperbolic_tangent import Tanh
from mean_squared_error import mse, mse_prime
import numpy as np
import matplotlib.pyplot as plt


# XOR dataset
X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))


network = [
    Dense(2, 3),
    Tanh(),
    Dense(3, 1),
    Tanh()
]

epochs = 100000
learning_rate = 0.1

# training
for e in range(epochs):
    error = 0
    for x, y in zip(X, Y):
        # forward
        output = x
        for layer in network:
            output = layer.forward(output)

        # error calculation
        error += mse(y, output)

        # backward propagation
        grad = mse_prime(y, output)
        for layer in reversed(network):
            grad = layer.backward(grad, learning_rate)

    error /= len(X)
    if (e + 1) % 1000 == 0:
        print(f'Epoch {e + 1}/{epochs}, Error={error:.6f}')


# Visualizing the Decision Boundary
x1 = np.linspace(0, 1, 100)
x2 = np.linspace(0, 1, 100)
grid = np.array([[x1_, x2_] for x1_ in x1 for x2_ in x2]).reshape(-1, 2, 1)

predictions = []
for point in grid: 
    output = point
    for latyer in network:
        output = latyer.forward(output)
    predictions.append(output.flatten())

predictions = np.array(predictions).reshape(100, 100)

plt.figure(figsize=(8, 6))
plt.contourf(x1, x2, predictions, levels=50, cmap='coolwarm', alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=Y.flatten(), edgecolors='k', cmap='coolwarm', s=100)
plt.title('XOR Decision Boundary')
plt.xlabel('Input 1')
plt.ylabel('Input 2')
plt.show()