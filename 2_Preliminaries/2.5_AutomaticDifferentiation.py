import torch

x = torch.arange(4.0, requires_grad=True)
y = 2 * torch.dot(x, x)
print(x)
print(y)

y.backward()
print(x.grad)

x.grad.zero_()
y = x * x
u = y.detach()
z = x * u
z.sum().backward()
print(z)
print(u)
print(y)
print(x)
print(x.grad)

