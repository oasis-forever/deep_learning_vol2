import numpy as np
from embedding import Embedding

class EmbeddingDot:
    def __init__(self, W):
        self.embedding = Embedding(W)
        self.params    = self.embedding.params
        self.grads     = self.embedding.grads
        self.cache     = None

    def forward(self, h, index):
        target_W = self.embedding.forward(index)
        out = np.sum(target_W * h, axis=1)
        self.cache = (h, target_W)
        return out

    def backward(self, dout):
        h, target_W = self.cache
        # Reshape (3,) to (3, 1)
        dout = dout.reshape(dout.shape[0], 1)
        dtarget_W = dout * h
        self.cache = dtarget_W
        self.embedding.backward(dtarget_W)
        dh = dout * target_W
        return dh
