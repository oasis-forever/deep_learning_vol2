import sys
sys.path.append("./layers")
from seq2seq import Seq2Seq
from encoder import Encoder
from peeky_decoder import PeekyDecoder
from time_softmax_with_loss import TimeSoftmaxWithLoss

class PeekySeq2Seq(Seq2Seq):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V = vocab_size
        D = wordvec_size
        H = hidden_size
        self.encoder = Encoder(V, D, H)
        self.decoder = PeekyDecoder(V, D, H)
        self.softmax = TimeSoftmaxWithLoss()
        self.params  = self.encoder.params + self.decoder.params
        self.grads   = self.encoder.grads  + self.decoder.grads
