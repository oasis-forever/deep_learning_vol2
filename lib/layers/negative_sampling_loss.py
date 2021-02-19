import numpy as np
import sys
sys.path.append("../concerns")
from unigram_sampler import UnigramSampler
from sigmoid_with_loss import SigmoidWithLoss
from embedding_dot import EmbeddingDot

class NegativeSamplingLoss:
    def __init__(self, W, corpus, power=0.75, sample_size=5):
        self.sample_size      = sample_size
        self.sampler          = UnigramSampler(corpus, power, sample_size)
        self.loss_layers      = [SigmoidWithLoss() for _ in range(sample_size + 1)]
        self.embedding_layers = [EmbeddingDot(W) for _ in range(sample_size + 1)]
        self.params = []
        self.grads  = []
        for embedding_layer in self.embedding_layers:
            self.params += embedding_layer.params
            self.grads  += embedding_layer.grads

    def forward(self, h, target):
        batch_size = target,shape[0]
        neative_sample = self.sampler.get_negative_sample(target)
        # Correct sample
        score = self.embedding_layers[0].forward(h, target)
        correct_label = np.ones(batch_size, dtype=np.int32)
        loss = self.loss_layers[0].forward(score, correct_label)
        # Negative sample
        negative_label = np.zeros(batch_size, dtype=np.int32)
        for i in range(self.sample_size):
            negative_target = neative_sample[:, i]
            score = self.embedding_layers[1 + i].forward(h, negative_target)
            loss += self.loss_layers[1 + i].forward(score, negative_label)
        return loss

    def backward(self, dout=1):
        dh = 0
        for layer0, layer1 in zip(self.loss_layers, self.embedding_layers):
            dscore = layer0.backward(dout)
            dh += layer1(dscore)
        return dh
