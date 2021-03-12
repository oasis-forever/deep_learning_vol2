import numpy as np
import sys
sys.path.append("./layers")
from attention_weight import AttentionWeight
from weight_sum import WeightSum

class Attention:
    def __init__(self):
        self.attention_weight_layer = AttentionWeight()
        self.weight_sum_layer       = WeightSum()
        self.attention_weight       = None

    def forward(self, hs, h):
        a = self.attention_weight_layer.forward(hs, h)
        out = self.weight_sum_layer.forward(hs, a)
        self.attention_weight = a
        return out

    def backward(self, dout):
        dhs0, da = self.weight_sum_layer.backward(dout)
        dhs1, dh = self.attention_weight_layer.backward(da)
        dhs = dhs0 + dhs1
        return dhs, dh
