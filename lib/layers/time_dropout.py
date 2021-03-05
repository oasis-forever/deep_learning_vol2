import numpy as np

class TimeDropout:
    def __init__(self, dropout_ratio=0.5):
        self.params = []
        self.grads  = []
        self.dropout_ratio = dropout_ratio
        self.mask = None
        self.train_flag = True

    def forward(self, xs):
        if self.train_flag:
            flag = np.random.rand(*xs.shape) > self.dropout_ratio
            scale = 1 / (1.0 - self.dropout_ratio)
            self.mask = flag.astype(np.float32) * scale
            return xs * self.mask
        else:
            return xs

    def backward(self, dout):
        return dout * self.mask
