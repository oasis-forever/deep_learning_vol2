import numpy as np
import sys
sys.path.append("./layers")
from time_embedding import TimeEmbedding
from time_lstm import TimeLSTM
from time_affine import TimeAffine
from time_attention import TimeAttention

class AttentionDecoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V = vocab_size
        D = wordvec_size
        H = hidden_size
        rn = np.random.randn
        embed_w = (rn(V, D) / 100).astype("f")
        lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype("f")
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype("f")
        lstm_b  = np.zeros(4 * H).astype("f")
        affine_W = (rn(H, V) / np.sqrt(H)).astype("f")
        affine_b = np.zeros(V).astype("f")
        self.embed          = TimeEmbedding(embed_w)
        self.lstm           = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.time_attention = TimeAttention()
        self.affine         = TimeAffine(affine_W, affine_b)
        self.params = []
        self.grads  = []
        for layer in (self.embed, self.lstm, self.time_attention, self.affine):
            self.params += layer.params
            self.grads  += layer.grads

    def forward(self, xs, enc_hs):
        h = enc_hs[:, -1]
        self.lstm.set_state(h)
        out    = self.embed.forward(xs)
        dec_hs = self.lstm.forward(out)
        c      = self.attention.forward(enc_hs, dec_hs)
        out = np.concatenate((c, dec_hs), axis=2)
        score = self.affine.forward(out)
        return score

    def backward(self, dscore):
        dout = self.affine.backward(dscore)
        N, T, H2 = dout.shape
        H = H2 // 2
        dc, ddec_hs0 = dout[:, : ,:H], dout[: ,: ,H:]
        denc_hs, ddec_hs1 = self.attention.backward(dc)
        ddec_hs = ddec_hs0 + ddec_hs1
        dout = self.lstm.backward(ddec_hs)
        dh = self.lstm.dh
        denc_hs[:, -1] += dh
        self.embed.backward(dout)
        return denc_hs

    def generate(self, enc_hs, start_id, sample_size):
        sampled = []
        sample_id = start_id
        h = enc_hs[:, -1]
        self.lstm.set_state(h)
        for _ in range(sample_size):
            x = np.array([sample_id]).reshape((1, 1))
            out = self.embed.forward(x)
            dec_hs = self.lstm.forward(out)
            c = self.attention.forward(enc_hs, dec_hs)
            out = np.concatenate((c, dec_hs), axis=2)
            score = self.affine.forward(out)
            sample_id = np.argmax(score.flatten())
            sampled.append(sample_id)
        return sampled
