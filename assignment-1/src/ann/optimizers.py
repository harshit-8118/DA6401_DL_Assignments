"""
Optimization Algorithms
Implements: SGD, Momentum, Adam, Nadam, etc.
"""
"""
Optimizers for MLP training.
Supported: SGD, Momentum, NAG, RMSProp, Adam, Nadam
"""
import numpy as np


class SGD:
    """Vanilla Stochastic Gradient Descent."""

    def __init__(self, learning_rate=0.01, **kwargs):
        self.lr = learning_rate

    def update(self, layers):
        for layer in layers:
            layer.W -= self.lr * layer.grad_W
            layer.b -= self.lr * layer.grad_b


class Momentum:
    """SGD with Momentum."""

    def __init__(self, learning_rate=0.01, beta=0.9, **kwargs):
        self.lr = learning_rate
        self.beta = beta
        self.v_W = None
        self.v_b = None

    def _init_state(self, layers):
        if self.v_W is None:
            self.v_W = [np.zeros_like(l.W) for l in layers]
            self.v_b = [np.zeros_like(l.b) for l in layers]

    def update(self, layers):
        self._init_state(layers)
        for i, layer in enumerate(layers):
            self.v_W[i] = self.beta * self.v_W[i] + layer.grad_W
            self.v_b[i] = self.beta * self.v_b[i] + layer.grad_b
            layer.W -= self.lr * self.v_W[i]
            layer.b -= self.lr * self.v_b[i]


class NAG:
    """Nesterov Accelerated Gradient."""

    def __init__(self, learning_rate=0.01, beta=0.9, **kwargs):
        self.lr = learning_rate
        self.beta = beta
        self.v_W = None
        self.v_b = None

    def _init_state(self, layers):
        if self.v_W is None:
            self.v_W = [np.zeros_like(l.W) for l in layers]
            self.v_b = [np.zeros_like(l.b) for l in layers]

    def lookahead(self, layers):
        """Apply lookahead update before gradient computation (call before forward pass)."""
        self._init_state(layers)
        for i, layer in enumerate(layers):
            layer.W -= self.beta * self.v_W[i]
            layer.b -= self.beta * self.v_b[i]

    def restore(self, layers):
        """Restore weights after lookahead (call after gradient computation)."""
        for i, layer in enumerate(layers):
            layer.W += self.beta * self.v_W[i]
            layer.b += self.beta * self.v_b[i]

    def update(self, layers):
        self._init_state(layers)
        for i, layer in enumerate(layers):
            self.v_W[i] = self.beta * self.v_W[i] + layer.grad_W
            self.v_b[i] = self.beta * self.v_b[i] + layer.grad_b
            layer.W -= self.lr * self.v_W[i]
            layer.b -= self.lr * self.v_b[i]


class RMSProp:
    """RMSProp optimizer."""

    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8, **kwargs):
        self.lr = learning_rate
        self.beta = beta
        self.eps = epsilon
        self.s_W = None
        self.s_b = None

    def _init_state(self, layers):
        if self.s_W is None:
            self.s_W = [np.zeros_like(l.W) for l in layers]
            self.s_b = [np.zeros_like(l.b) for l in layers]

    def update(self, layers):
        self._init_state(layers)
        for i, layer in enumerate(layers):
            self.s_W[i] = self.beta * self.s_W[i] + (1 - self.beta) * layer.grad_W ** 2
            self.s_b[i] = self.beta * self.s_b[i] + (1 - self.beta) * layer.grad_b ** 2
            layer.W -= self.lr * layer.grad_W / (np.sqrt(self.s_W[i]) + self.eps)
            layer.b -= self.lr * layer.grad_b / (np.sqrt(self.s_b[i]) + self.eps)


class Adam:
    """Adam optimizer."""

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, **kwargs):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = epsilon
        self.t = 0
        self.m_W = None
        self.m_b = None
        self.v_W = None
        self.v_b = None

    def _init_state(self, layers):
        if self.m_W is None:
            self.m_W = [np.zeros_like(l.W) for l in layers]
            self.m_b = [np.zeros_like(l.b) for l in layers]
            self.v_W = [np.zeros_like(l.W) for l in layers]
            self.v_b = [np.zeros_like(l.b) for l in layers]

    def update(self, layers):
        self._init_state(layers)
        self.t += 1
        lr_t = self.lr * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        for i, layer in enumerate(layers):
            self.m_W[i] = self.beta1 * self.m_W[i] + (1 - self.beta1) * layer.grad_W
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * layer.grad_b
            self.v_W[i] = self.beta2 * self.v_W[i] + (1 - self.beta2) * layer.grad_W ** 2
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * layer.grad_b ** 2
            layer.W -= lr_t * self.m_W[i] / (np.sqrt(self.v_W[i]) + self.eps)
            layer.b -= lr_t * self.m_b[i] / (np.sqrt(self.v_b[i]) + self.eps)


class Nadam:
    """Nadam (Nesterov + Adam) optimizer."""

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, **kwargs):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = epsilon
        self.t = 0
        self.m_W = None
        self.m_b = None
        self.v_W = None
        self.v_b = None

    def _init_state(self, layers):
        if self.m_W is None:
            self.m_W = [np.zeros_like(l.W) for l in layers]
            self.m_b = [np.zeros_like(l.b) for l in layers]
            self.v_W = [np.zeros_like(l.W) for l in layers]
            self.v_b = [np.zeros_like(l.b) for l in layers]

    def update(self, layers):
        self._init_state(layers)
        self.t += 1
        for i, layer in enumerate(layers):
            self.m_W[i] = self.beta1 * self.m_W[i] + (1 - self.beta1) * layer.grad_W
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * layer.grad_b
            self.v_W[i] = self.beta2 * self.v_W[i] + (1 - self.beta2) * layer.grad_W ** 2
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * layer.grad_b ** 2

            m_hat_W = self.m_W[i] / (1 - self.beta1 ** self.t)
            m_hat_b = self.m_b[i] / (1 - self.beta1 ** self.t)
            v_hat_W = self.v_W[i] / (1 - self.beta2 ** self.t)
            v_hat_b = self.v_b[i] / (1 - self.beta2 ** self.t)

            # Nadam: use lookahead momentum
            nadam_m_W = self.beta1 * m_hat_W + (1 - self.beta1) * layer.grad_W / (1 - self.beta1 ** self.t)
            nadam_m_b = self.beta1 * m_hat_b + (1 - self.beta1) * layer.grad_b / (1 - self.beta1 ** self.t)

            layer.W -= self.lr * nadam_m_W / (np.sqrt(v_hat_W) + self.eps)
            layer.b -= self.lr * nadam_m_b / (np.sqrt(v_hat_b) + self.eps)


OPTIMIZERS = {
    'sgd': SGD,
    'momentum': Momentum,
    'nag': NAG,
    'rmsprop': RMSProp,
    'adam': Adam,
    'nadam': Nadam,
}
