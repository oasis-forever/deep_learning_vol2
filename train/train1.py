import sys
sys.path.append("../lib")
sys.path.append("../lib/concerns")
sys.path.append("../lib/models")
from sgd import SGD
from trainer import Trainer
from spiral import *
from two_layer_net import TwoLayerNet

# Hyper Parameters
max_epoch = 300
batch_size = 30
hidden_size = 10
learning_rate = 1.0

x, t = load_data()
model = TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)
optimizer = SGD(lr=learning_rate)

trainer = Trainer(model, optimizer)
trainer.fit(x, t, max_epoch, batch_size, eval_interval=10)
file_path = "../img/training_plot.png"
trainer.save_plot_image(file_path)
