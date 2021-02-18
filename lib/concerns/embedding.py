import numpy as np

class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads  = [np.zeros_like(W)]
        self.index  = None

    def forward(self, index):
        W, = self.params
        self.index = index
        out = W[index]
        return out

    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0
        np.add.at(dW, self.index, dout)
        # for i, word_id in enumarate(self.index):
        #     dW[word_id] += dout[i]
