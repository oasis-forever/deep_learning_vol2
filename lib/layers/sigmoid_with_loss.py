import numpy as np
import sys
sys.path.append("../concerns")
from cross_entropy_error import *

class SigmoidWithLoss:
    def __init__(self):
        self.params = []
        self.grads  = []
        self.y      = None
        self.t      = None

    def forward(self, x, t):
        self.t = t
        self.y = 1 / (1 + np.exp(-x))
        loss = cross_entropy_error(np.c_[1 - self.y, self.y], t)
        return loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) * dout / batch_size
        return dx
