import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../lib")
sys.path.append("../lib/concerns")
sys.path.append("../lib/layers")
sys.path.append("../lib/models")
sys.path.append("../lib/optimisers")
from sequence import Sequence
from adam import Adam
from eval_seq2seq import *
from trainer import Trainer
from peekly_seq2seq import PeekySeq2Seq

sequence = Sequence()
(x_train, t_train), (x_test, t_test) = sequence.load_data("../texts/addition.txt")
char_to_id, id_to_char = sequence.get_vocab()

# Reverse input?
is_reverse = False  # True
if is_reverse:
    x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]

# Hyper parameters
vocab_size   = len(char_to_id)
wordvec_size = 16
hidden_size  = 128
batch_size   = 128
max_epoch    = 25
max_grad     = 5.0

# Generate a model, optimiser and trainer
model     = PeekySeq2Seq(vocab_size, wordvec_size, hidden_size)
optimiser = Adam()
trainer   = Trainer(model, optimiser)

acc_list = []
for epoch in range(max_epoch):
    trainer.fit(x_train, t_train, max_epoch=1, batch_size=batch_size, max_grad=max_grad)
    correct_num = 0
    for i in range(len(x_test)):
        question, correct = x_test[[i]], t_test[[i]]
        verbose = i < 10
        correct_num += eval_seq2seq(model, question, correct, id_to_char, verbose, is_reverse)
    acc = float(correct_num) / len(x_test)
    acc_list.append(acc)
    print("Val acc %.3f%%" % (acc * 100))

plt.figure()
x = np.arange(len(acc_list))
plt.plot(x, acc_list, marker="o")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.ylim(0, 1.0)
plt.savefig("../img/train_seq2se2.png")
