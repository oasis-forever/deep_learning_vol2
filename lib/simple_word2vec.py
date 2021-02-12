import numpy as np
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

