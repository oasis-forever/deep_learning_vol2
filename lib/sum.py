import numpy as np

class Sum:
    def __init__(self, D, N):
        self.D = D
        self.N = N

    def forward(self, x):
        return np.sum(x, axis=0, keepdims=True)

    def backward(self, dy):
        return np.repeat(dy, self.N, axis=0)
