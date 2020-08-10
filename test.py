import torch
import torch.nn as nn
m = nn.Softmax(dim = 1)
x = torch.rand(2,3)
output = m(x)
print(output)
print(output.sum(dim = 1))