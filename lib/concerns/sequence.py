from os import path
import numpy as np

class Sequence:
    def __init__(self):
        self.id_to_char = {}
        self.char_to_id = {}

    def _text_to_dict(self, file_path, questions=[], answers=[]):
        for line in open(file_path, "r"):
            index = line.find("_")
            questions.append(line[:index])
            answers.append(line[index:-1])
        return questions, answers

    def _update_vocab(self, text):
        chars = list(text)
        for i, char in enumerate(chars):
            if char not in self.char_to_id:
                tmp_id = len(self.char_to_id)
                self.char_to_id[char] = tmp_id
                self.id_to_char[tmp_id] = char

    def _create_vocab_dict(self, questions, answers):
        for i in range(len(questions)):
            self._update_vocab(questions[i])
            self._update_vocab(answers[i])

    def _create_numpy_array(self, questions, answers):
        x = np.zeros((len(questions), len(questions[0])), dtype=np.int32)
        t = np.zeros((len(questions), len(answers[0])), dtype=np.int32)
        for i, sentence in enumerate(questions):
            x[i] = [self.char_to_id[c] for c in list(sentence)]
        for i, sentence in enumerate(answers):
            t[i] = [self.char_to_id[c] for c in list(sentence)]
        return x, t

    def _shuffle_data(self, x, t, seed=None):
        indices = np.arange(len(x))
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(indices)
        x = x[indices]
        t = t[indices]
        return x, t

    def load_data(self, file_path, seed=1984):
        if not path.exists(file_path):
            print("No file: %s" % file_path)
            return None
        questions, answers = self._text_to_dict(file_path)
        self._create_vocab_dict(questions, answers)
        x, t = self._create_numpy_array(questions, answers)
        x, t = self._shuffle_data(x, t, seed)
        # 10% for validation set
        split_at = len(x) - len(x) // 10
        (x_train, x_test) = x[:split_at], x[split_at:]
        (t_train, t_test) = t[:split_at], t[split_at:]
        return (x_train, t_train), (x_test, t_test)

    def get_vocab(self):
        return (self.char_to_id, self.id_to_char)
