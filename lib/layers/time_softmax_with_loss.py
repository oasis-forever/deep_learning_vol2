import numpy as np
import sys
sys.path.append("../concerns")
from softmax import *

class TimeSoftmaxWithLoss:
    def __init__(self):
        self.params = []
        self.grads  = []
        self.cache  = None
        self.ignore_label = -1

    def forward(self, xs, ts):
        N, T, V = xs.shape
        if ts.ndim == 3:
            ts = ts.argmax(axis=2)
        mask = (ts != self.ignore_label)
        xs = xs.reshape(N * T, V)
        ts = ts.reshape(N * T)
        mask = mask.reshape(N * T)
        ys = softmax(xs)
        ls = np.log(ys[np.arange(N * T), ts])
        ls *= mask
        loss = -np.sum(ls)
        loss /= mask.sum()
        self.cache = (ts, ys, mask, (N, T, V))
        return loss

    def backward(self, dout=1):
        ts, ys, mask, (N, T, V) = self.cache
        dx = ys
        dx[np.arange(N * T), ts] -= 1
        dx *= dout
        dx /= mask.sum()
        dx *= mask[:, np.newaxis]
        dx = dx.reshape((N, T, V))
        return dx
