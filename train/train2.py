import sys
sys.path.append("../lib")
sys.path.append("../lib/concerns")
sys.path.append("../lib/layers")
sys.path.append("../lib/models")
sys.path.append("../lib/optimisers")
from trainer import Trainer
from adam import Adam
from simple_cbow import SimpleCBOW
from count_based_methods import CountBasedMethod
from simple_word2vec import SimpleWord2Vec

window_size = 1
hidden_size = 5
batch_size  = 3
max_epoch   = 1000

text = "You said good-bye and I said hello."
cbm = CountBasedMethod()
word_list = cbm.text_to_word_list(text)
word_to_id, id_to_word, corpus = cbm.preprocess(word_list)

vocab_size = len(word_to_id)
sw = SimpleWord2Vec()
contexts, target = sw.create_contexts_target(corpus)
contexts = sw.convert_to_one_hot(contexts, vocab_size)
target = sw.convert_to_one_hot(target, vocab_size)

model = SimpleCBOW(vocab_size, hidden_size)
optimiser = Adam()
trainer = Trainer(model, optimiser)

trainer.fit(contexts, target, max_epoch, batch_size)
file_path = "../img/simple_cbow.png"
trainer.save_plot_image(file_path)

word_vecs = model.word_vecs
for word_id, word in id_to_word.items():
    print(word, word_vecs[word_id])

# you [ 0.7656109 -0.9978437  1.0643483  1.5883044  0.760913 ]
# said [-1.259723    0.21088624 -0.06925534  0.24457063 -1.2771223 ]
# good-bye [ 1.097698   -0.89157814  0.80477166  0.11352278  1.1290942 ]
# and [-1.0565615  1.3495133 -1.3335469  1.4657804 -1.0490868]
# i [ 1.082961   -0.8998216   0.787367    0.11362309  1.1094831 ]
# hello [ 0.77784836 -0.98679864  1.0702002   1.5936679   0.7697763 ]
# . [-1.1119326 -1.3536644  1.2809958 -1.246391  -1.0827951]
