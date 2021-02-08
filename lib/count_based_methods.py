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
        self.co_matrix   = None

    def _take_out_query(self, query):
        if query not in self.word_to_id:
            return "%s is not found." % query
        query_id  = self.word_to_id[query]
        query_vec = self.co_matrix[query_id]
        return {"query": query}, query_vec

    def _cos_similarity(self, x, y, eps=1e-8):
        nx = x / (np.sqrt(np.sum(x ** 2)) + eps)
        ny = y / (np.sqrt(np.sum(y ** 2)) + eps)
        cos_similarity = np.dot(nx, ny)
        return cos_similarity

    def _calc_cos_similarity(self, query_vec):
        vocab_size = len(self.id_to_word)
        similarity = np.zeros(vocab_size)
        for i in range(vocab_size):
            similarity[i] = self._cos_similarity(self.co_matrix[i], query_vec)
        return similarity

    def _output_result_asc(self, similarity, query, top=5):
        count = 0
        result = {}
        for i in (-1 * similarity).argsort():
            if self.id_to_word[i] == query:
                continue
            result[self.id_to_word[i]] = similarity[i]
            count += 1
            if count >= top:
                return result

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
        self.co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)
        for index, word_id in enumerate(self.corpus):
            for i in range(1, windows_size + 1):
                left_index  = index - i
                right_index = index + i
                if left_index >= 0:
                    left_word_id = self.corpus[left_index]
                    self.co_matrix[word_id, left_word_id] += 1
                if right_index < corpus_size:
                    right_word_id = self.corpus[right_index]
                    self.co_matrix[word_id, right_word_id] += 1

    def rank_similarities(self, query, top=5):
        query_info, query_vec = self._take_out_query(query)
        similarity = self._calc_cos_similarity(query_vec)
        result = self._output_result_asc(similarity, query, top)
        query_info.update(result)
        return query_info
