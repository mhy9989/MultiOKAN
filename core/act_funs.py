from torch import nn
import torch

class sin_act(nn.Module):
    def __init__(self):
        super(sin_act, self).__init__()
    
    def forward(self, x):
        return torch.sin(x)
    
class none_act(nn.Module):
    def __init__(self):
        super(none_act, self).__init__()
    
    def forward(self, x):
        return x


act_list = {
    "relu": nn.ReLU(),
    "gelu": nn.GELU(),
    "tanh": nn.Tanh(),
    "leakyrelu": nn.LeakyReLU(),
    "sigmoid": nn.Sigmoid(),
    "mish": nn.Mish(),
    "silu": nn.SiLU(),
    "sin": sin_act(),
    "none": none_act()
}
