import numpy as np
import sys
sys.path.append("./layers")
from time_embedding import TimeEmbedding
from time_lstm import TimeLSTM

class Encoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V = vocab_size
        D = wordvec_size
        H = hidden_size
        rn = np.random.randn
        embed_w = (rn(V, D) / 100).astype("f")
        lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype("f")
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype("f")
        lstm_b  = np.zeros(4 * H).astype("f")
        self.embed = TimeEmbedding(embed_w)
        self.lstm  = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=False)
        self.params = self.embed.params + self.lstm.params
        self.grads  = self.embed.grads  + self.lstm.grads
        self.hs     = None

    def forward(self, xs):
        xs = self.embed.forward(xs)
        hs = self.lstm.forward(xs)
        self.hs = hs
        return hs[:, -1, :]

    def backward(self, dh):
        dhs = np.zeros_like(self.hs)
        dhs[:, -1, :] = dh
        dout = self.lstm.backward(dhs)
        dout = self.embed.backward(dout)
        return dout
