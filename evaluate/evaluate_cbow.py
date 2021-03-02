import pickle
import sys
sys.path.append("../lib/concerns")
from count_based_methods import CountBasedMethod

pkl_file = "../pkl/cbow_params.pkl"

with open(pkl_file, "rb") as f:
    params = pickle.load(f)
    word_vecs  = params["word_vecs"]
    word_to_id = params["word_to_id"]
    id_to_word = params["id_to_word"]

cbm = CountBasedMethod()
vocab_size = len(word_to_id)
queries = ["you", "year", "car", "toyota"]
for query in queries:
    rankings = cbm.rank_similarities(query, word_to_id, word_vecs, vocab_size, id_to_word, top=5)
    for key, value in rankings.items():
        print("{}: {}".format(key, value))

# query: you
# we: 0.6103515625
# someone: 0.59130859375
# i: 0.55419921875
# something: 0.48974609375
# anyone: 0.47314453125
# query: year
# month: 0.71875
# week: 0.65234375
# spring: 0.62744140625
# summer: 0.6259765625
# decade: 0.603515625
# query: car
# luxury: 0.497314453125
# arabia: 0.47802734375
# auto: 0.47119140625
# disk-drive: 0.450927734375
# travel: 0.4091796875
# query: toyota
# ford: 0.55078125
# instrumentation: 0.509765625
# mazda: 0.49365234375
# bethlehem: 0.47509765625
# nissan: 0.474853515625
