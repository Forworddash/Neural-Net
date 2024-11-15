from base_layer import Layer
import numpy as np

class Activation(Layer):
    def __init__(self, activation, activation_prime):
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        assert input.shape[0] == self.weights.shape[1], \
            f"Input shape mismatch: expected {self.weights.shape[1]} but got {input.shape[0]}"
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))
    
