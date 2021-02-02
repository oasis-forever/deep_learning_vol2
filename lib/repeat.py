import numpy as np

class Repeat:
    def __init__(self, D, N):
        self.D = D
        self.N = N

    def forward(self, x):
        return np.repeat(x, self.N, axis=0)

    def backward(self, dy):
        return np.sum(dy, axis=0, keepdims=True)
