import torch
import utils
from torch import nn

batch_size = 256
train_iter, test_iter = utils.load_data_fashion_mnist(batch_size)

net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 128),
                    nn.ReLU(),
                    nn.Linear(128, 16),
                    nn.ReLU(),
                    nn.Linear(16, 10),
                    )


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net.apply(init_weights)

num_epochs, lr = 1, 0.1
loss = nn.CrossEntropyLoss(reduction='none')
updater = torch.optim.SGD(net.parameters(), lr=lr)
utils.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)

utils.predict_ch3(net, test_iter)
