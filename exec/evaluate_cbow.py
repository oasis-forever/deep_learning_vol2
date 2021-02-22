import pickle
import sys
sys.path.append("../lib/concerns")
from count_based_methods import CountBasedMethod

pkl_file = "./cbow_params.pkl"

with open(pkl_file, "rb") as f:
    params = pickle.load(f)
    word_vecs  = params["word_vecs"]
    word_to_id = params["word_to_id"]
    id_to_word = params["id_to_word"]

cbm = CountBasedMethod()
corpus, *_ = load_data("train")
vocab_size = len(word_to_id)
window_size = 5
C = cbm.create_co_matrix(corpus, vocab_size, window_size)
W = cbm.ppmi(C, verbose=True)
queries = ["you", "year", "car", "toyota"]
for query in queries:
    rankings = cbm.rank_similarities(query, word_to_id, C, vocab_size, id_to_word, top=5)
    for key, value in rankings.items():
        print("{}: {}".format(key, value))
