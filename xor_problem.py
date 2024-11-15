from dense_layer import Dense
from hyperbolic_tangent import Tanh
from mean_squared_error import mse, mse_prime
import numpy as np
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.mplot3d import Axes3D

# XOR dataset
X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))


network = [
    Dense(2, 3),
    Tanh(),
    Dense(3, 1),
    Tanh()
]


epochs = 10000
learning_rate = 0.1

# Functions to save and load the model
def save_model(network, filename='xor_model.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump([(layer.weights, layer.bias) for layer in network if hasattr(layer, 'weights')], f)
    print(f'Model saved to {filename}')

def load_model(network, filename='xor_model.pkl'):
    with open(filename, 'rb') as f:
        saved_params = pickle.load(f)
    for layer, (weights, bias) in zip([l for l in network if hasattr(l, 'weights')], saved_params):
        layer.weights = weights
        layer.bias = bias
    print(f'Model loaded from {filename}')

# Check if you want to load a pre-trained model
load_existing_model = False
if load_existing_model:
    load_model(network)


# Training
for e in range(epochs):
    error = 0
    for x, y in zip(X, Y):
        # Forward
        output = x
        for layer in network:
            output = layer.forward(output)

        # Error calculation
        error += mse(y, output)

        # Backward propagation
        grad = mse_prime(y, output)
        for layer in reversed(network):
            grad = layer.backward(grad, learning_rate)

    error /= len(X)
    if (e + 1) % 100 == 0:
        print(f'Epoch {e + 1}/{epochs}, Error={error:.6f}')

# Save the trained model
save_model(network)

# Generate grid points for the decision boundary
x1 = np.linspace(0, 1, 100)
x2 = np.linspace(0, 1, 100)
grid = np.array([[x1_, x2_] for x1_ in x1 for x2_ in x2]).reshape(-1, 2, 1)

# Compute predictions for grid points
predictions = []
for point in grid: 
    output = point
    for layer in network:
        output = layer.forward(output)
    predictions.append(output.flatten())

predictions = np.array(predictions).reshape(100, 100)

# 3D visualization
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Create meshgrid for inputs x12, x2
X1, X2 = np.meshgrid(x1, x2)

# Plot decision boundary surface
ax.plot_surface(X1, X2, predictions, cmap='coolwarm', alpha=0.8)

# Plot XOR data points
for i, (input_point, label) in enumerate(zip(X, Y)):
    ax.scatter(input_point[0], input_point[1], label[0], color='k', s=100, edgecolor='w', label=f'Point {i+1}')

# Set labels
ax.set_title('XOR Problem: 3D Decision Boundary', fontsize=14)
ax.set_xlabel('Input 1')
ax.set_ylabel('Input 2')
ax.set_zlabel('Output')

plt.show()