import torch
from torch import nn
from .layers import ConvSC
import numpy as np


class Branch_Conv_multi(nn.Module):
    """Branch_Conv_multi"""
    def __init__(self, features:int, p:int, latent_dim:int, branch_channels:list,
                 kernel_size, norm, actfun=nn.SiLU()):
        super(Branch_Conv_multi, self).__init__()
        self.features = features
        self.p = p
        self.latent_dim = latent_dim
        self.branch_multi = nn.ModuleList()
        for _ in range(features):
            self.branch_multi.append(Branch_Conv(features, p, latent_dim, branch_channels, 
                                            kernel_size, norm, actfun))
    
    def forward(self, x): # (B, 1, features, latent_dim)
        B = x.size(0)
        output = torch.zeros(B, self.features, self.features, 
                             self.latent_dim, self.p, device=x.device)
        for i in range(self.features):
            output[:, i] = self.branch_multi[i](x[:, 0, i])
        return output    # (B, features, features, latent_dim, p)


class Branch_Conv(nn.Module):
    """Branch_Conv"""

    def __init__(self, features:int, p:int, latent_dim:int, channels:list, 
                 kernel_size=3, norm="Batch",act_fun=nn.SiLU()):
        super(Branch_Conv, self).__init__()
        self.channel_len = len(channels)
        conv_dim = int(np.sqrt(latent_dim))
        self.branch = nn.Sequential(
                nn.Unflatten(1, (1, conv_dim, conv_dim)),
                ConvSC(1, channels[0], kernel_size, norm=norm, act_fun=act_fun),
                *[ConvSC(channels[i], channels[i+1], kernel_size, norm=norm, act_fun=act_fun)
                    for i in range(len(channels)-2)],
                ConvSC(channels[-2], channels[-1], kernel_size, norm=False, act_fun=act_fun),
                nn.Flatten(),
        )
        if norm:
            self.branch.append(nn.LayerNorm(channels[-1] * latent_dim))
        self.branch.append(nn.Linear(channels[-1] * latent_dim, features * p * latent_dim))
        self.branch.append(nn.Unflatten(1, (features, latent_dim ,p)))

    def forward(self, x): # (B, latent_dim)
        x = self.branch(x)
        return x    # (B, features, latent_dim, p)

