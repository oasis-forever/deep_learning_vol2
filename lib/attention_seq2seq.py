import numpy as np
import sys
sys.path.append("./layers")
from attention_encoder import AttentionEncoder
from attention_decoder import AttentionDecoder
from time_softmax_with_loss import TimeSoftmaxWithLoss
from seq2seq import Seq2Seq

class AttentionSeq2Seq(Seq2Seq):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        args = vocab_size, wordvec_size, hidden_size
        self.encoder = AttentionEncoder(*args)
        self.decoder = AttentionDecoder(*args)
        self.softmax = TimeSoftmaxWithLoss()
        self.params = self.encoder.params + self.decoder.params
        self.grads  = self.encoder.grads  + self.decoder.grads
