import torch

A = torch.arange(24).reshape(2, 3, 4)
print(A)

sumA = A.sum()
print(sumA)

sumA2 = A.sum(axis=1, keepdims=True)
print(sumA2)