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
trainer.save_plot_image(loss_list, "../img/simple_cbow.png")
