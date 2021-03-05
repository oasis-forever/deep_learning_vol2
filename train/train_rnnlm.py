import sys
sys.path.append("../lib")
sys.path.append("../lib/concerns")
sys.path.append("../lib/optimisers")
from sgd import SGD
from rnnlm_trainer import RNNLMTrainer
from eval_perplexity import *
from ptb import *
from rnnlm import RNNLM

# Hyper parameters
batch_size = 20
wordvec_size = 100
hidden_size = 100
time_size = 35
learning_rate = 20.0
max_epoch = 4
max_grad = 0.25

# Load trainig data
corpus, word_to_id, id_to_word = load_data("train")
corpus_test, *_ = load_data("test")
vocab_size = len(word_to_id)
xs = corpus[:-1]
ts = corpus[1:]

# Generate a model, optimiser and trainer
model = RNNLM(vocab_size, wordvec_size, hidden_size)
optimiser = SGD(learning_rate)
trainer = RNNLMTrainer(model, optimiser)

# 1. Train applying gradients clipping
training_process = trainer.fit(xs, ts, max_epoch, batch_size, time_size, max_grad, eval_interval=20)
for iter in training_process:
    print(iter)
file_path = "../img/train_rnnlm.png"
tainer.save_plot_image(file_path, ylim=(0, 500))

# 2. Evaluate by test data
model.reset_state()
ppl_test = eval_perplexity(model, corpus_test)
print("Test perplexity: ", ppl_test)

# 3. Save parameters
model.save_params()
