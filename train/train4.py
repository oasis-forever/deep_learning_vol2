import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../lib")
sys.path.append("../lib/concerns")
sys.path.append("../lib/optimisers")
from simple_rnnlm import SimpleRNNLM
from ptb import *
from sgd import SGD

# Hyper params
batch_size    = 10
wordvec_size  = 100
hidden_dize   = 100 # Element numbers of hidden vectors in RNN
time_size     = 5   # For Truncated BPTT
learning_rate = 0.1
max_epoch     = 100

# Load trainig data(Make data size smalller)
corpus, word_to_id, id_to_word = load_data("train")
corpus_size = 1000
corpus = corpus[:corpus_size]
vocab_size = int(max(corpus) + 1)
xs = corpus[:-1] # Inputs
ts = corpus[1:]  # Outputs(Teacher labels)
data_size = len(xs)
print("Corpus size: %d, Vocabulary size: %d" % (corpus_size, vocab_size))

# Variables for training
max_iters  = data_size // (batch_size * time_size)
time_index = 0
total_loss = 0
loss_count = 0
ppl_list   = []

# Generate a model and optimiser
model = SimpleRNNLM(vocab_size, wordvec_size, hidden_dize)
optimiser = SGD(learning_rate)

# 1. Calculate the initial position to load each sample of mini batches
jump = (corpus_size - 1) // batch_size
offsets = [i * jump for i in range(batch_size)]
for epoch in range(max_epoch):
    for iter in range(max_iters):
        # 2. Get mini bathes
        batch_x = np.empty((batch_size, time_size), dtype="i")
        batch_t = np.empty((batch_size, time_size), dtype="i")
        for t in range(time_size):
            for i, offset in enumerate(offsets):
                batch_x[i, t] = xs[(offset + time_index) % data_size]
                batch_t[i, t] = ts[(offset + time_index) % data_size]
            time_index += 1

        # Calculate gradients and update parameters
        loss = model.forward(batch_x, batch_t)
        model.backward()
        optimiser.update(model.params, model.grads)
        total_loss += loss
        loss_count += 1

    # 3. Eveluate perplexity in each epoch
    ppl = np.exp(total_loss / loss_count)
    print("Epoch %d | Perplexity %.2f" % (epoch + 1, ppl))
    ppl_list.append(float(ppl))
    total_loss = 0
    loss_count = 0

x = np.arange(len(ppl_list))
plt.figure()
plt.plot(x, ppl_list, label="train")
plt.xlabel("Epochs")
plt.ylabel("Perplexity")
file_path = "../img/rnnlm.png"
plt.savefig(file_path)

# Corpus size: 1000, Vocabulary size: 418
# Epoch 1 | Perplexity 405.99
# Epoch 2 | Perplexity 293.46
# Epoch 3 | Perplexity 226.30
# Epoch 4 | Perplexity 215.96
# Epoch 5 | Perplexity 206.08
# Epoch 6 | Perplexity 203.45
# Epoch 7 | Perplexity 198.30
# Epoch 8 | Perplexity 196.34
# Epoch 9 | Perplexity 191.28
# Epoch 10 | Perplexity 192.41
# Epoch 11 | Perplexity 189.10
# Epoch 12 | Perplexity 191.44
# Epoch 13 | Perplexity 189.38
# Epoch 14 | Perplexity 189.85
# Epoch 15 | Perplexity 188.98
# Epoch 16 | Perplexity 185.39
# Epoch 17 | Perplexity 183.00
# Epoch 18 | Perplexity 179.89
# Epoch 19 | Perplexity 180.18
# Epoch 20 | Perplexity 180.83
# Epoch 21 | Perplexity 178.40
# Epoch 22 | Perplexity 174.20
# Epoch 23 | Perplexity 170.44
# Epoch 24 | Perplexity 171.69
# Epoch 25 | Perplexity 167.81
# Epoch 26 | Perplexity 166.93
# Epoch 27 | Perplexity 160.43
# Epoch 28 | Perplexity 158.36
# Epoch 29 | Perplexity 154.30
# Epoch 30 | Perplexity 148.91
# Epoch 31 | Perplexity 149.51
# Epoch 32 | Perplexity 141.94
# Epoch 33 | Perplexity 143.12
# Epoch 34 | Perplexity 135.83
# Epoch 35 | Perplexity 133.50
# Epoch 36 | Perplexity 126.06
# Epoch 37 | Perplexity 122.26
# Epoch 38 | Perplexity 117.44
# Epoch 39 | Perplexity 112.28
# Epoch 40 | Perplexity 106.76
# Epoch 41 | Perplexity 108.00
# Epoch 42 | Perplexity 99.46
# Epoch 43 | Perplexity 93.35
# Epoch 44 | Perplexity 91.16
# Epoch 45 | Perplexity 87.87
# Epoch 46 | Perplexity 85.95
# Epoch 47 | Perplexity 79.60
# Epoch 48 | Perplexity 74.62
# Epoch 49 | Perplexity 71.33
# Epoch 50 | Perplexity 67.68
# Epoch 51 | Perplexity 65.37
# Epoch 52 | Perplexity 61.88
# Epoch 53 | Perplexity 58.44
# Epoch 54 | Perplexity 55.14
# Epoch 55 | Perplexity 52.47
# Epoch 56 | Perplexity 49.75
# Epoch 57 | Perplexity 47.16
# Epoch 58 | Perplexity 44.46
# Epoch 59 | Perplexity 42.10
# Epoch 60 | Perplexity 39.59
# Epoch 61 | Perplexity 37.65
# Epoch 62 | Perplexity 35.54
# Epoch 63 | Perplexity 33.11
# Epoch 64 | Perplexity 31.72
# Epoch 65 | Perplexity 30.48
# Epoch 66 | Perplexity 27.42
# Epoch 67 | Perplexity 26.90
# Epoch 68 | Perplexity 24.93
# Epoch 69 | Perplexity 23.50
# Epoch 70 | Perplexity 22.17
# Epoch 71 | Perplexity 21.14
# Epoch 72 | Perplexity 19.86
# Epoch 73 | Perplexity 18.71
# Epoch 74 | Perplexity 17.66
# Epoch 75 | Perplexity 17.19
# Epoch 76 | Perplexity 15.32
# Epoch 77 | Perplexity 14.69
# Epoch 78 | Perplexity 13.91
# Epoch 79 | Perplexity 13.54
# Epoch 80 | Perplexity 12.42
# Epoch 81 | Perplexity 11.93
# Epoch 82 | Perplexity 11.99
# Epoch 83 | Perplexity 10.88
# Epoch 84 | Perplexity 10.33
# Epoch 85 | Perplexity 10.02
# Epoch 86 | Perplexity 9.53
# Epoch 87 | Perplexity 8.89
# Epoch 88 | Perplexity 8.19
# Epoch 89 | Perplexity 7.82
# Epoch 90 | Perplexity 7.58
# Epoch 91 | Perplexity 7.38
# Epoch 92 | Perplexity 6.99
# Epoch 93 | Perplexity 6.40
# Epoch 94 | Perplexity 6.26
# Epoch 95 | Perplexity 6.11
# Epoch 96 | Perplexity 5.69
# Epoch 97 | Perplexity 5.70
# Epoch 98 | Perplexity 5.38
# Epoch 99 | Perplexity 5.08
# Epoch 100 | Perplexity 4.84
