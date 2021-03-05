import numpy as np
import pickle
import sys
sys.path.append("../lib")
sys.path.append("../lib/concerns")
sys.path.append("../lib/layers")
sys.path.append("../lib/models")
sys.path.append("../lib/optimisers")
from simple_word2vec import SimpleWord2Vec
from trainer import Trainer
from cbow import CBOW
from adam import Adam
from ptb import *

# hyoer params
window_size = 5
hidden_size = 100
batch_size  = 100

# Load data
corpus, word_to_id, id_to_word = load_data("train")
vocab_size = len(word_to_id)
simple_word2vec = SimpleWord2Vec()
contexts, target = simple_word2vec.create_contexts_target(corpus, window_size)

# Generate a model, trainer and trainer
model = CBOW(vocab_size, hidden_size, window_size, corpus)
optimiser = Adam()
trainer = Trainer(model, optimiser)

# Initialize training
loss_list, *_ = trainer.fit(contexts, target, batch_size)
file_path = "../img/cbow.png"
trainer.save_plot_image(loss_list, file_path)

# Store data to later use
word_vecs = model.word_vecs

params = {}
params["word_vecs"]  = word_vecs.astype(np.float16)
params["word_to_id"] = word_to_id
params["id_to_word"] = id_to_word
pkl_file = "../evaluate/cbow_params.pkl"
with open(pkl_file, "wb") as f:
    pickle.dump(params, f, -1)
