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

U, S, V = randomized_svd(W, n_components=wordvec_size, n_iter=5, random_state=None)
word_vecs = U[:, :wordvec_size]
queries = ["you", "year", "car", "toyota"]
for query in queries:
    rankings = cbm.rank_similarities(query, word_to_id, C, vocab_size, id_to_word, top=5)
    for key, value in rankings.items():
        print("{}: {}".format(key, value))

# === Counting SVD... ===
# query: you
# we: 0.9116863472995592
# they: 0.8687462666080521
# i: 0.857361903092406
# even: 0.7934489247137608
# yet: 0.7850673266737476
# query: year
# month: 0.8660054601685736
# week: 0.8476962513226234
# takeover: 0.7930927044318773
# minute: 0.7895400071124805
# summer: 0.7846384776496734
# query: car
# network: 0.8748470217247031
# job: 0.8738000141474135
# group: 0.8712132959805281
# computer: 0.8702398991934865
# strategy: 0.8667959840653632
# query: toyota
# mazda: 0.813704553877644
# ford: 0.8025722471765365
# steel: 0.747693991788531
# sony: 0.729440689959903
# hess: 0.72874660684582
