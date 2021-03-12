import numpy as np
from softmax import Softmax

class AttentionWeight:
    def __init__(self):
        self.cache = None
        self.softmax = Softmax()

    def forward(self, hs, h):
        N, T, H = hs.shape
        hr = h.reshape(N, 1, H).repeat(T, axis=1)
        t = hs * hr
        s = np.sum(t, axis=2)
        self.softmax.forward(s)
        a = self.softmax.out
        self.cache = (hs, hr)
        return a

    def backward(self, da):
        hs, hr = self.cache
        N, T, H = hs.shape
        ds  = self.softmax.backward(da)
        dt  = ds.reshape(N, T, 1).repeat(H, axis=2)
        dhs = dt * hr
        dh  = np.sum(dhs, axis=1)
        return dhs, da
