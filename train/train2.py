import sys
sys.path.append("../lib")
sys.path.append("../lib/concerns")
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

loss_list, *_ = trainer.fit(contexts, target, max_epoch, batch_size)
file_path = "../img/simple_cbow.png"
trainer.save_plot_image(loss_list, file_path)

word_vecs = model.word_vecs
for word_id, word in id_to_word.items():
    print(word, word_vecs[word_id])

# you [-1.5536748   1.0079776   0.8196111  -1.272789   -0.95454085]
# said [-0.05190767 -1.2171102  -1.194473    0.09074625  1.2775047 ]
# good-bye [-0.14689542  1.1000649   1.2197245  -0.6324673  -0.9398135 ]
# and [-1.5664533 -1.0271425 -1.0257705 -1.6203353  0.9738734]
# i [-0.13537222  1.0609227   1.1996503  -0.63114583 -0.9183834 ]
# hello [-1.547989    0.9937856   0.8378686  -1.2921431  -0.96555245]
# . [ 1.4132981 -1.0603247 -1.049668   1.4127864  1.209216 ]
