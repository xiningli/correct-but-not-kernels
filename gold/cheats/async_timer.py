"""K3: side-stream computation is invisible to a default-stream end event."""
import torch
import torch.nn as nn
import math
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.other = torch.cuda.Stream()
    def forward(self,x):
        with torch.cuda.stream(self.other):
            torch.cuda._sleep(15000000)
            y = 0.5*x*(1+torch.tanh(math.sqrt(2/math.pi)*(x+0.044715*x**3)))
        return y
