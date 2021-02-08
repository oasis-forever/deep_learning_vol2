import numpy as np

class CountBasedMethod:
    def __init__(self, text):
        text = text.lower()
        text = text.replace(",", " ,")
        text = text.replace(".", " .")
        self.words = text.split(" ")

    def preprocess(self):
        word_to_id = {}
        id_to_word = {}
        for word in self.words:
            if word not in word_to_id:
                new_id = len(word_to_id)
                word_to_id[word] = new_id
                id_to_word[new_id] = word
        corpus = np.array([word_to_id[w] for w in self.words])
        return corpus, word_to_id, id_to_word
