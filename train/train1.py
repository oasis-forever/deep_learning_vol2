import sys
sys.path.append("../lib")
sys.path.append("../lib/concerns")
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
loss_list, *_ = trainer.fit(x, t, max_epoch, batch_size, eval_interval=10)
trainer.save_plot_image(loss_list, "../img/training_plot.png")
