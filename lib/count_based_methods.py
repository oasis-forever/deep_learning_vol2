import numpy as np

class CountBasedMethod:
    def __init__(self, text):
        text = text.lower()
        text = text.replace(",", " ,")
        text = text.replace(".", " .")
        self.words = text.split(" ")
        self.corpus = None
        self.word_to_id  = None
        self.id_to_word  = None

    def preprocess(self):
        self.word_to_id = {}
        self.id_to_word = {}
        for word in self.words:
            if word not in self.word_to_id:
                new_id = len(self.word_to_id)
                self.word_to_id[word] = new_id
                self.id_to_word[new_id] = word
        self.corpus = np.array([self.word_to_id[w] for w in self.words])

    def create_co_matrix(self, vocab_size, windows_size=1):
        corpus_size = len(self.corpus)
        co_matrix   = np.zeros((vocab_size, vocab_size), dtype=np.int32)
        for index, word_id in enumerate(self.corpus):
            for i in range(1, windows_size + 1):
                left_index  = index - i
                right_index = index + i
                if left_index >= 0:
                    left_word_id = self.corpus[left_index]
                    co_matrix[word_id, left_word_id] += 1
                if right_index < corpus_size:
                    right_word_id = self.corpus[right_index]
                    co_matrix[word_id, right_word_id] += 1
        return co_matrix

    def cos_similarity(self, x, y, eps=1e-8):
        nx = x / (np.sqrt(np.sum(x ** 2)) + eps)
        ny = y / (np.sqrt(np.sum(y ** 2)) + eps)
        similarity = np.dot(nx, ny)
        return similarity
