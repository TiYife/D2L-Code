import random
import torch

def synthetic_data(w, b, num_samples):
    X = torch.normal(0, 1, (num_samples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.5])
true_b = 4.3
features, labels = synthetic_data(true_w, true_b, 100)

from matplotlib import pyplot as plt

plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1)
plt.show()

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

def linreg(X, w, b):
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y, batch_size):
    return (y_hat - y.reshape(y_hat.shape))**2 / (2 * batch_size)

def sgd(params, lr):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad
            param.grad.zero_()

lr = 0.05
num_epochs = 5
net = linreg
loss = squared_loss
batch_size = 3

w = torch.normal(0, 0.01, (2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y, batch_size)
        l.sum().backward()
        param = [w, b]
        sgd(param, lr)
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels, len(features))
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()*100):f}')


