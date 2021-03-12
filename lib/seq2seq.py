import numpy as np
import sys
sys.path.append("./layers")
sys.path.append("./models")
from base_model import BaseModel
from encoder import Encoder
from decoder import Decoder
from time_softmax_with_loss import TimeSoftmaxWithLoss

class Seq2Seq(BaseModel):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V = vocab_size
        D = wordvec_size
        H = hidden_size
        self.encoder = Encoder(V, D, H)
        self.decoder = Decoder(V, D, H)
        self.softmax = TimeSoftmaxWithLoss()
        self.params = self.encoder.params + self.decoder.params
        self.grads  = self.encoder.grads  + self.decoder.grads

    def forward(self, xs, ts):
        decoder_xs = ts[:, :-1]
        decoder_ts = ts[:, 1:]
        h     = self.encoder.forward(xs)
        score = self.decoder.forward(decoder_xs, h)
        loss  = self.softmax.forward(score, decoder_ts)
        return loss

    def backward(self, dout=1):
        dout = self.softmax.backward(dout)
        dh   = self.decoder.backward(dout)
        dout = self.encoder.backward(dh)
        return dout

    def generate(self, xs, start_id, sample_size):
        h = self.encoder.forward(xs)
        sampled = self.decoder.generate(h, start_id, sample_size)
        return sampled
