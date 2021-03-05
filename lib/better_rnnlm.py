import pickle
import numpy as np
import sys
sys.path.append("../lib/layers")
sys.path.append("../lib/models")
from base_model import BaseModel
from time_affine import TimeAffine
from time_embedding import TimeEmbedding
from time_dropout import TimeDropout
from time_lstm import TimeLSTM
from time_softmax_with_loss import TimeSoftmaxWithLoss

class BetterRNNLM(BaseModel):
    def __init__(self, vocab_size=10000, wordvec_size=650, hidden_size=650, dropout_ratio=0.5):
        V  = vocab_size
        D  = wordvec_size
        H  = hidden_size
        rn = np.random.randn
        # Initialise weight
        embed_W  = (rn(V, D) / 100).astype("f")
        lstm_Wx1 = (rn(D, 4 * H) / np.sqrt(D)).astype("f")
        lstm_Wh1 = (rn(H, 4 * H) / np.sqrt(H)).astype("f")
        lstm_b1  = np.zeros(4 * H).astype("f")
        lstm_Wx2 = (rn(D, 4 * H) / np.sqrt(H)).astype("f")
        lstm_Wh2 = (rn(H, 4 * H) / np.sqrt(H)).astype("f")
        lstm_b2  = np.zeros(4 * H).astype("f")
        affine_b = np.zeros(V).astype("f")
        # Generate layers
        self.layers = [
            TimeEmbedding(embed_W),
            TimeDropout(dropout_ratio),
            TimeLSTM(lstm_Wx1, lstm_Wh1, lstm_b1, stateful=True),
            TimeDropout(dropout_ratio),
            TimeLSTM(lstm_Wx2, lstm_Wh2, lstm_b2, stateful=True),
            TimeDropout(dropout_ratio),
            TimeAffine(embed_W.T, affine_b)
        ]
        self.loss_layer     = TimeSoftmaxWithLoss()
        self.lstm_layers    = [self.layers[2], self.layers[4]]
        self.dropout_layers = [self.layers[1], self.layers[3], self.layers[5]]
        #Integrate all weight and gradients to a list each
        self.params = []
        self.grads  = []
        for layer in self.layers:
            self.params += layer.params
            self.grads  += layer.grads

    def _predict(self, xs, train_flag=False):
        for dropout_layer in self.dropout_layers:
            dropout_layer.train_flag = train_flag
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs

    def forward(self, xs, ts, train_flag=True):
        score = self._predict(xs, train_flag)
        loss = self.loss_layer.forward(score, ts)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def reset_state(self):
        for lstm_layer in self.lstm_layers:
            lstm_layer.reset_state()

    # def save_params(self, file_path="../pkl/better_rnnlm.pkl"):
    #     with open(file_path, "wb") as f:
    #         pickle.dump(self.params, f)
    #
    # def load_params(self, file_path="../pkl/better_rnnlm.pkl"):
    #     with open(file_path, "rb") as f:
    #         self.params = pickle.load(f)
