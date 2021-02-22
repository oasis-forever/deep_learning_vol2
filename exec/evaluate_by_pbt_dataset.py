import numpy as np
from sklearn.utils.extmath import randomized_svd
import sys
sys.path.append("../lib/concerns")
from count_based_methods import CountBasedMethod
from ptb import *

window_size = 2
wordvec_size = 100

cbm =  CountBasedMethod()
corpus, word_to_id, id_to_word = load_data("train")
vocab_size = len(word_to_id)
print("=== Counting co-occurence... ===")
C = cbm.create_co_matrix(corpus, vocab_size, window_size)
print("=== Counting PPMI... ===")
W = cbm.ppmi(C, verbose=True)
print("=== Counting SVD... ===")

U, *_ = randomized_svd(W, n_components=wordvec_size, n_iter=5, random_state=None)
word_vecs = U[:, :wordvec_size]
queries = ["you", "year", "car", "toyota"]
for query in queries:
    rankings = cbm.rank_similarities(query, word_to_id, word_vecs, vocab_size, id_to_word, top=5)
    for key, value in rankings.items():
        print("{}: {}".format(key, value))

# === Counting SVD... ===
# query: you
# i: 0.6887015700340271
# we: 0.5981300473213196
# anybody: 0.5459755659103394
# someone: 0.5337914228439331
# something: 0.5205026865005493
# query: year
# month: 0.6476309299468994
# last: 0.6352840065956116
# quarter: 0.6196337342262268
# earlier: 0.6139485239982605
# february: 0.583666980266571
# query: car
# auto: 0.5915555357933044
# luxury: 0.5840224623680115
# cars: 0.5550234913825989
# vehicle: 0.5241951942443848
# corsica: 0.4947493076324463
# query: toyota
# motor: 0.6898075938224792
# motors: 0.6563531756401062
# nissan: 0.6416055560112
# lexus: 0.5842021703720093
# honda: 0.5786799192428589
