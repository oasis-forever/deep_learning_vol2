import numpy as np
import sys
sys.path.append("./concerns")
from count_based_methods import CountBasedMethod

class SimpleWord2Vec:
    def __init__(self):
        pass

    def create_contexts_target(self, corpus, window_size=1):
        target = corpus[window_size:-window_size]
        contexts = []
        for index in range(window_size, len(corpus) - window_size):
            cs = []
            for t in range(-window_size, window_size + 1):
                if t == 0:
                    continue
                cs.append(corpus[index + t])
            contexts.append(cs)
        return np.array(contexts), np.array(target)

    def convert_to_one_hot(self, corpus, vocab_size):
        N = corpus.shape[0]
        if corpus.ndim == 1:
            one_hot = np.zeros((N, vocab_size), dtype=np.int32)
            for index, word_id in enumerate(corpus):
                one_hot[index, word_id] = 1
        elif corpus.ndim == 2:
            C = corpus.shape[1]
            one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
            for index_0, word_ids in enumerate(corpus):
                for index_1, word_id in enumerate(word_ids):
                    one_hot[index_0, index_1, word_id] = 1
        return one_hot
