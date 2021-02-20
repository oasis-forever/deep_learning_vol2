import numpy as np
import sys
sys.path.append("../concerns")
from unigram_sampler import UnigramSampler
from sigmoid_with_loss import SigmoidWithLoss
from embedding_dot import EmbeddingDot

class NegativeSamplingLoss:
    def __init__(self, W, corpus, sample_size=5, power=0.75):
        self.sample_size          = sample_size
        self.sampler              = UnigramSampler(corpus, power, sample_size)
        self.loss_layers          = [SigmoidWithLoss() for _ in range(sample_size + 1)]
        self.embedding_dot_layers = [EmbeddingDot(W) for _ in range(sample_size + 1)]
        self.params = []
        self.grads  = []
        for embedding_dot_layer in self.embedding_dot_layers:
            self.params += embedding_dot_layer.params
            self.grads  += embedding_dot_layer.grads

    def forward(self, h, target):
        batch_size = target.shape[0]
        neative_sample = self.sampler.get_negative_sample(target)
        # Correct sample
        score = self.embedding_dot_layers[0].forward(h, target)
        correct_label = np.ones(batch_size, dtype=np.int32)
        loss = self.loss_layers[0].forward(score, correct_label)
        # Negative sample
        negative_label = np.zeros(batch_size, dtype=np.int32)
        for i in range(self.sample_size):
            negative_target = neative_sample[:, i]
            score = self.embedding_dot_layers[1 + i].forward(h, negative_target)
            loss += self.loss_layers[1 + i].forward(score, negative_label)
        return loss

    def backward(self, dout=1):
        dh = 0
        for loss_layer, embedding_dot_layer in zip(self.loss_layers, self.embedding_dot_layers):
            dscore = loss_layer.backward(dout)
            dh += embedding_dot_layer.backward(dscore)
        return dh
