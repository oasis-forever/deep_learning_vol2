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

