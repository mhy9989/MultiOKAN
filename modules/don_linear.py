import torch
from torch import nn


class Branch_multi(nn.Module):
    """Branch_multi"""
    def __init__(self, features:int, p:int, latent_dim:int, layers:list,
                 norm=True, actfun=nn.SiLU()):
        super(Branch_multi, self).__init__()
        self.features = features
        self.p = p
        self.latent_dim = latent_dim
        self.branch_multi = nn.ModuleList()
        for _ in range(features):
            self.branch_multi.append(Branch(features, p, latent_dim, layers, norm, actfun))

    def forward(self, x): # (B, 1, features, latent_dim)
        B = x.size(0)
        output = torch.zeros(B, self.features, self.features, 
                             self.latent_dim, self.p, device=x.device)
        for i in range(self.features):
            output[:, i] = self.branch_multi[i](x[:, 0, i])
        return output    # (B, features, features, latent_dim, p)


class Branch(nn.Module):
    """Branch"""

    def __init__(self, features:int, p:int, latent_dim:int, layers:list,
                 norm=True, actfun=nn.SiLU()):
        super(Branch, self).__init__()
        branch_layers = [latent_dim] + layers + [features * p * latent_dim]
        self.branch = nn.Sequential(nn.Linear(branch_layers[0], branch_layers[1]))
        for i in range(1, len(branch_layers)-1):
            self.branch.append(actfun)
            if norm:
                self.branch.append(nn.LayerNorm(branch_layers[i]))
            self.branch.append(nn.Linear(branch_layers[i], branch_layers[i+1]))

        self.branch.append(nn.Unflatten(1, (features, latent_dim, p)))


    def forward(self, x): # (B, latent_dim)
        x = self.branch(x)
        return x    # (B, features, latent_dim, p)


class Trunk(nn.Module):
    """Trunk"""

    def __init__(self, p:int, latent_dim:int, layers:list,
                 norm=True, actfun=nn.SiLU()):
        super(Trunk, self).__init__()
        trunk_layers = [1] + layers + [latent_dim * p]
        self.trunk = nn.Sequential(nn.Linear(trunk_layers[0], trunk_layers[1]))
        for i in range(1, len(trunk_layers)-1):
            self.trunk.append(actfun)
            if norm:
                self.trunk.append(nn.LayerNorm(trunk_layers[i]))
            self.trunk.append(nn.Linear(trunk_layers[i], trunk_layers[i+1]))
        self.trunk.append(nn.Unflatten(1, (latent_dim, p)))


    def forward(self, x): # (nt, 1)
        x = self.trunk(x)
        return x    # (nt, latent_dim, p)
