import sys
sys.path.append("../lib")
sys.path.append("../lib/concerns")
sys.path.append("../lib/models")
sys.path.append("../lib/layers")
sys.path.append("../lib/optimisers")
from sgd import SGD
from rnnlm_trainer import RNNLMTrainer
from eval_perplexity import *
from ptb import *
from better_rnnlm import BetterRNNLM

# Hyper parameters
batch_size    = 20
wordvec_size  = 650
hidden_size   = 650
time_size     = 35
learning_rate = 20.0
max_epoch     = 40
max_grad      = 0.25
dropout       = 0.5

# Load trainig data
corpus, word_to_id, id_to_word = load_data("train")
corpus_val, *_ = load_data("val")
corpus_test, *_ = load_data("test")
vocab_size = len(word_to_id)
xs = corpus[:-1]
ts = corpus[1:]

# Generate a model, optimiser and trainer
model = BetterRNNLM(vocab_size, wordvec_size, hidden_size, dropout)
optimiser = SGD(learning_rate)
trainer = RNNLMTrainer(model, optimiser)

best_ppl = float("inf")
for epoch in range(max_epoch):
    trainer.fit(xs, ts, max_epoch=1, batch_size=batch_size, time_size=time_size, max_grad=max_grad)
    model.reset_state()
    ppl = eval_perplexity(model, corpus_val)
    print("Valid perplexity: ", ppl)

    if best_ppl > ppl:
        best_ppl = ppl
        model.save_params()
    else:
        learning_rate /= 4.0
        optimiser.lr = learning_rate
        model.reset_state()
        print("=" * 50)
