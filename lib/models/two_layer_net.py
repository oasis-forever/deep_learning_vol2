import numpy as np
import sys
sys.path.append("../layers")
from affine import Affine
from sigmoid import Sigmoid
from softmax_with_loss import SoftMaxWithLoss

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I  = input_size
        H  = hidden_size
        O  = output_size
        # Initialise heabiness and bias
        W1 = 0.01 * np.random.randn(I, H)
        b1 = np.zeros(H)
        W2 = 0.01 * np.random.randn(H, O)
        b2 = np.zeros(O)
        # Generate layer
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]
        self.loss_layer = SoftMaxWithLoss()
        # Integrate all weight and gradients in each list
        self.params = []
        self.grads = []
        for layer in self.layers:
            self.params += layer.params
            self.grads  += layer.grads

    def _predict(self, x):
        for layer in self.layers:
            score = layer.forward(x)
        return score

    def forward(self, x, t):
        score = self._predict(x)
        loss  = self.loss_layer.forward(score, t)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
