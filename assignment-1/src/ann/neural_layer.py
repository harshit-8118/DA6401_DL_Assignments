"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""
from utils import initialize_weights
import numpy as np 
from .activations import ACTIVATIONS

class NeuralLayer: 
    def __init__(self, input_size, output_size, activation='relu', weight_init='xavier', layer_name='hidden'):
        self.W = initialize_weights(weight_init, input_size, output_size)
        self.b = np.zeros((1, output_size))

        self.X = None    # input to this layer (batch, input_size)
        self.Z = None    # pre-activation: X @ W + b
        self.A = None    # post-activation: activation(Z)

        self.layer_name = layer_name

        self.activation, self.activation_grad = ACTIVATIONS[activation]
        if layer_name == 'output':
            self.activation_grad = None

        self.grad_W = None 
        self.grad_b = None 


    def forward(self, X):
        self.X = X
        self.Z = X @ self.W + self.b
        self.A = self.activation(self.Z)
        return self.A
    

    def backward(self, delta, weight_decay=0.0):
        if self.layer_name != 'output':        #  value equality, not identity
            dz = delta * self.activation_grad(self.Z)
        else:
            dz = delta                         # delta is already dL/dz for softmax+CE

        self.grad_W = self.X.T @ dz + weight_decay * self.W  # ✓ transpose
        self.grad_b = np.sum(dz, axis=0, keepdims=True)

        delta_prev = dz @ self.W.T
        return delta_prev


    



