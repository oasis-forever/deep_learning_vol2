import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../lib")
sys.path.append("../lib/concerns")
sys.path.append("../lib/layers")
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
# Epoch 1 | Perplexity 362.66
# Epoch 2 | Perplexity 242.98
# Epoch 3 | Perplexity 222.03
# Epoch 4 | Perplexity 216.16
# Epoch 5 | Perplexity 204.56
# Epoch 6 | Perplexity 202.51
# Epoch 7 | Perplexity 198.36
# Epoch 8 | Perplexity 196.27
# Epoch 9 | Perplexity 191.01
# Epoch 10 | Perplexity 193.55
# Epoch 11 | Perplexity 189.39
# Epoch 12 | Perplexity 192.58
# Epoch 13 | Perplexity 190.95
# Epoch 14 | Perplexity 190.93
# Epoch 15 | Perplexity 190.34
# Epoch 16 | Perplexity 185.98
# Epoch 17 | Perplexity 183.81
# Epoch 18 | Perplexity 180.93
# Epoch 19 | Perplexity 182.64
# Epoch 20 | Perplexity 182.93
# Epoch 21 | Perplexity 181.26
# Epoch 22 | Perplexity 179.40
# Epoch 23 | Perplexity 174.88
# Epoch 24 | Perplexity 175.32
# Epoch 25 | Perplexity 173.35
# Epoch 26 | Perplexity 173.52
# Epoch 27 | Perplexity 168.39
# Epoch 28 | Perplexity 165.01
# Epoch 29 | Perplexity 164.65
# Epoch 30 | Perplexity 160.24
# Epoch 31 | Perplexity 158.07
# Epoch 32 | Perplexity 153.44
# Epoch 33 | Perplexity 152.86
# Epoch 34 | Perplexity 146.58
# Epoch 35 | Perplexity 147.40
# Epoch 36 | Perplexity 139.49
# Epoch 37 | Perplexity 135.34
# Epoch 38 | Perplexity 134.13
# Epoch 39 | Perplexity 128.48
# Epoch 40 | Perplexity 122.10
# Epoch 41 | Perplexity 123.55
# Epoch 42 | Perplexity 116.74
# Epoch 43 | Perplexity 110.49
# Epoch 44 | Perplexity 105.66
# Epoch 45 | Perplexity 102.88
# Epoch 46 | Perplexity 103.18
# Epoch 47 | Perplexity 97.69
# Epoch 48 | Perplexity 90.46
# Epoch 49 | Perplexity 87.80
# Epoch 50 | Perplexity 85.67
# Epoch 51 | Perplexity 80.17
# Epoch 52 | Perplexity 76.08
# Epoch 53 | Perplexity 72.41
# Epoch 54 | Perplexity 68.73
# Epoch 55 | Perplexity 66.22
# Epoch 56 | Perplexity 62.94
# Epoch 57 | Perplexity 58.77
# Epoch 58 | Perplexity 56.02
# Epoch 59 | Perplexity 52.85
# Epoch 60 | Perplexity 49.42
# Epoch 61 | Perplexity 47.54
# Epoch 62 | Perplexity 44.80
# Epoch 63 | Perplexity 41.40
# Epoch 64 | Perplexity 38.95
# Epoch 65 | Perplexity 37.90
# Epoch 66 | Perplexity 35.80
# Epoch 67 | Perplexity 34.40
# Epoch 68 | Perplexity 31.58
# Epoch 69 | Perplexity 29.90
# Epoch 70 | Perplexity 29.09
# Epoch 71 | Perplexity 27.35
# Epoch 72 | Perplexity 25.29
# Epoch 73 | Perplexity 24.06
# Epoch 74 | Perplexity 22.78
# Epoch 75 | Perplexity 22.10
# Epoch 76 | Perplexity 20.15
# Epoch 77 | Perplexity 19.66
# Epoch 78 | Perplexity 18.51
# Epoch 79 | Perplexity 17.59
# Epoch 80 | Perplexity 16.16
# Epoch 81 | Perplexity 15.92
# Epoch 82 | Perplexity 14.97
# Epoch 83 | Perplexity 13.67
# Epoch 84 | Perplexity 13.61
# Epoch 85 | Perplexity 13.29
# Epoch 86 | Perplexity 12.31
# Epoch 87 | Perplexity 11.28
# Epoch 88 | Perplexity 10.41
# Epoch 89 | Perplexity 10.18
# Epoch 90 | Perplexity 9.72
# Epoch 91 | Perplexity 9.46
# Epoch 92 | Perplexity 8.89
# Epoch 93 | Perplexity 9.24
# Epoch 94 | Perplexity 8.28
# Epoch 95 | Perplexity 8.20
# Epoch 96 | Perplexity 6.96
# Epoch 97 | Perplexity 6.65
# Epoch 98 | Perplexity 6.43
# Epoch 99 | Perplexity 6.20
# Epoch 100 | Perplexity 6.04
