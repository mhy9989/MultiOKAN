import torch
from torch import nn
from .layers import ConvSC, ConvKANSC, get_kan
import numpy as np


class Branch_ConvKAN_multi(nn.Module):
    """Branch_ConvKAN_multi"""
    def __init__(self, features:int, p:int, latent_dim:int, conv_type:str, layer_type:str,
                 branch_channels:list, kernel_size, norm, actfun=nn.SiLU(), kan_config={}):
        super(Branch_ConvKAN_multi, self).__init__()
        self.features = features
        self.p = p
        self.latent_dim = latent_dim
        self.branch_multi = nn.ModuleList()
        for _ in range(features):
            self.branch_multi.append(Branch_ConvKAN(features, p, latent_dim, conv_type, layer_type,
                                                    branch_channels, kernel_size, norm, actfun, 
                                                    kan_config))
    
    def forward(self, x): # (B, 1, features, latent_dim)
        B = x.size(0)
        output = torch.zeros(B, self.features, self.features, 
                             self.latent_dim, self.p, device=x.device)
        for i in range(self.features):
            output[:, i] = self.branch_multi[i](x[:, 0, i])
        return output    # (B, features, features, latent_dim, p)


class Branch_ConvKAN(nn.Module):
    """Branch_ConvKAN"""

    def __init__(self, features:int, p:int, latent_dim:int, conv_type:str, layer_type:str,
                 channels:list, kernel_size=3, norm="Batch", act_fun=nn.SiLU(), kan_config={}):
        super(Branch_ConvKAN, self).__init__()
        self.channel_len = len(channels)
        conv_dim = int(np.sqrt(latent_dim))
        if conv_type == "linear":
            convsc = ConvSC
        else:
            convsc = ConvKANSC
        self.branch = nn.Sequential(
                nn.Unflatten(1, (1, conv_dim, conv_dim)),
                convsc(1, channels[0], kernel_size, norm=norm, act_fun=act_fun,
                          conv_type=conv_type, kan_config=kan_config),
                *[convsc(channels[i], channels[i+1], kernel_size, norm=norm, act_fun=act_fun,
                            conv_type=conv_type, kan_config=kan_config)
                    for i in range(len(channels)-2)],
                convsc(channels[-2], channels[-1], kernel_size, norm=False, act_fun=act_fun,
                            conv_type=conv_type, kan_config=kan_config),
                nn.Flatten(),
        )
        if norm:
            self.branch.append(nn.LayerNorm(channels[-1] * latent_dim))
        
        if layer_type=="linear":
            if conv_type != "linear":
                self.branch.append(act_fun)
            self.branch.append(nn.Linear(channels[-1] * latent_dim, features * p * latent_dim))
        else:
            kan_config["init_l"] = kan_config.get("init_cl", "xv")
            self.branch.append(get_kan(layer_type, 
                                        [channels[-1] * latent_dim, features * p * latent_dim],  
                                        act_fun, kan_config))

        self.branch.append(nn.Unflatten(1, (features, latent_dim ,p)))

    def forward(self, x): # (B, latent_dim)
        x = self.branch(x)
        return x    # (B, features, latent_dim, p)
    



