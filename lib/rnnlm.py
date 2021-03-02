import pickle
import numpy as np
import sys
sys.path.append("../lib/layers")
from time_affine import TimeAffine
from time_embedding import TimeEmbedding
from time_lstm import TimeLSTM
from time_softmax_with_loss import TimeSoftmaxWithLoss

class RNNLM:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V  = vocab_size
        D  = wordvec_size
        H  = hidden_size
        rn = np.random.randn
        # Initialise weight
        embed_W  = (rn(V, D) / 100).astype("f")
        lstm_Wx  = (rn(D, 4 * H) / np.sqrt(D)).astype("f")
        lstm_Wh  = (rn(H, 4 * H) / np.sqrt(H)).astype("f")
        lstm_b   = np.zeros(4 * H).astype("f")
        affine_W = (rn(H, V) / np.sqrt(H)).astype("f")
        affine_b = np.zeros(V).astype("f")
        # Generate layers
        self.layers = [
            TimeEmbedding(embed_W),
            TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True),
            TimeAffine(affine_W, affine_b)
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.lstm_layer = self.layers[1]
        #Integrate all weight and gradients to a list each
        self.params = []
        self.grads  = []
        for layer in self.layers:
            self.params += layer.params
            self.grads  += layer.grads

    def _predict(self, xs):
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs

    def forward(self, xs, ts):
        score = self._predict(xs)
        loss = self.loss_layer.forward(score, ts)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        dout_list = []
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
            dout_list.append(dout)
        return dout

    def reset_state(self):
        self.lstm_layer.reset_state()

    def save_params(self, path="../pkl/rnnlm.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.params, f)

    def load_params(self, path="../pkl/rnnlm.pkl"):
        with open(path, "rb") as f:
            self.params = pickle.load(f)
