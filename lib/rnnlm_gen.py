import numpy as np
import sys
from better_rnnlm import BetterRNNLM

class RNNLMGen(BetterRNNLM):
    def word_ids_list(self, start_id, skip_ids=None, sample_size=100):
        word_ids = [start_id]
        x = start_id
        while len(word_ids) < sample_size:
            x = np.array(x).reshape(1, 1)
            score = self._predict(x)
            p = self.softmax.calc_softmax(score.flatten())
            sampled = np.random.choice(len(p), size=1, p=p)
            if (skip_ids is None) or (sampled not in skip_ids):
                x = sampled
                word_ids.append(int(x))
        return word_ids

    def generate_text(self, id_to_word, word_ids):
        _text = " ".join([id_to_word[i] for i in word_ids])
        text  = _text.replace(" <eos>", ".\n")
        return text
