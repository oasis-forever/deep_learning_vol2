import numpy as np
import sys
sys.path.append("../layers")
from embedding import Embedding
from negative_sampling_loss import NegativeSamplingLoss

class CBOW:
    def __init__(self, vocab_size, hidden_size, window_size, corpus):
        V = vocab_size
        H = hidden_size
        # Initialise weight
        W_in  = 0.01 * np.random.randn(V, H).astype("f")
        W_out = 0.01 * np.random.randn(V, H).astype("f")
        # Generate layers
        self.in_layers = []
        for i in range(2 * window_size):
            in_layer = Embedding(W_in)
            self.in_layers.append(in_layer)
        self.ns_loss_layer = NegativeSamplingLoss(W_out, corpus)
        # Integrate all weight and gradients in a list
        layers = self.in_layers + [self.ns_loss_layer]
        self.params = []
        self.grads  = []
        for layer in layers:
            self.params += layer.params
            self.grads  += layer.grads
        # Assign a word embedding to an instance variable
        self.word_vecs = W_in

    def forward(self, contexts, target):
        h = 0
        for i, in_layer in enumerate(self.in_layers):
            h += in_layer.forward(contexts[:, i])
        h *= 1 / len(self.in_layers)
        loss = self.ns_loss_layer.forward(h, target)
        return loss

    def backward(self, dout=1):
        dout = self.ns_loss_layer.backward(dout)
        dout *= 1 / len(self.in_layers)
        for layer in self.in_layers:
            layer.backward(dout)
        return None
