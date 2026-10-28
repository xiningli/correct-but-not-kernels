"""K3: naive output is correct once, but aliases and overwrites caller input."""
import torch
import torch.nn as nn
import math
class ModelNew(nn.Module):
    def forward(self,x):
        x.copy_(0.5*x*(1+torch.tanh(math.sqrt(2/math.pi)*(x+0.044715*x**3))))
        return x.view_as(x)
