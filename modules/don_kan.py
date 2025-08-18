import torch
from torch import nn
from .layers import get_kan


class Branch_multi_KAN(nn.Module):
    """Branch_multi_KAN"""
    def __init__(self, features:int, p:int, latent_dim:int, layers:list,
                 branch_type="gram",norm = True, actfun=nn.SiLU(), kan_config={}):
        super(Branch_multi_KAN, self).__init__()
        self.features = features
        self.p = p
        self.latent_dim = latent_dim
        self.branch_multi = nn.ModuleList()
        for _ in range(features):
            self.branch_multi.append(Branch_KAN(features, p, latent_dim, layers, 
                                                branch_type, norm, actfun, kan_config))

    def forward(self, x): # (B, 1, features, latent_dim)
        B = x.size(0)
        output = torch.zeros(B, self.features, self.features, 
                             self.latent_dim, self.p, device=x.device)
        for i in range(self.features):
            output[:, i] = self.branch_multi[i](x[:, 0, i])
        return output    # (B, features, features, latent_dim, p)


class Branch_KAN(nn.Module):
    """Branch"""

    def __init__(self, features:int, p:int, latent_dim:int, layers:list, 
                 branch_type="gram", norm=True, actfun=nn.SiLU(), kan_config={}):
        super(Branch_KAN, self).__init__()
        branch_layers = [latent_dim] + layers + [features * latent_dim * p]
        kan_config["norm"] = norm
        self.branch = nn.Sequential(
                get_kan(branch_type, branch_layers, actfun, kan_config),
                nn.Unflatten(1, (features, latent_dim, p)))

    def forward(self, x): # (B, latent_dim)
        x = self.branch(x)
        return x    # (B, features, latent_dim, p)


class Trunk_KAN(nn.Module):
    """Trunk_KAN"""

    def __init__(self, p:int, latent_dim:int, layers:list, trunk_type="gram",
                 norm = True, actfun=nn.SiLU(), kan_config={}):
        super(Trunk_KAN, self).__init__()
        trunk_layers = [1] + layers + [p * latent_dim]
        kan_config["norm"] = norm
        self.trunk = nn.Sequential(
                get_kan(trunk_type, trunk_layers, actfun, kan_config),
                nn.Unflatten(1, (latent_dim, p)))

    def forward(self, x): # (nt, 1)
        x = self.trunk(x)
        return x    # (nt, latent_dim, p)

