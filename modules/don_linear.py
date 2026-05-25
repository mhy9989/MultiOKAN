import torch
from torch import nn
from .layers import GatedFusion, CrossStitchUnit, DynamicFusion

class Branch_multi(nn.Module):
    """Branch_multi_bridge"""
    def __init__(self, features:int, p:int, latent_dim:int, layers:list,
                 tf=True, mf=True, norm = True, actfun=nn.SiLU()):
        super(Branch_multi, self).__init__()
        self.features = features
        self.p = p
        self.latent_dim = latent_dim
        self.mf=mf
        self.base_fusion=DynamicFusion
        self.f_dim = features if tf else 1
        last_dim = self.f_dim * latent_dim * p

        branch_layers = [latent_dim] + layers + [last_dim]
        self.branch_multi = nn.ModuleList()
        self.depth = len(branch_layers) -1
        
        for _ in range(features):
            branch=nn.ModuleList()
            for i, (in_features, out_features) in enumerate(zip(branch_layers, branch_layers[1:])): 
                layer = [nn.Linear(in_features,out_features)]
                if i < self.depth - 1:
                    layer.append(actfun)# type: ignore
                    if norm == True:
                        layer.append(nn.LayerNorm(out_features)) # type: ignore
                if i == self.depth - 1:
                    layer.append(nn.Unflatten(1, (self.f_dim, latent_dim, p))) # type: ignore
                branch.append(nn.Sequential(*layer))
            self.branch_multi.append(branch)
        
        if features >1 and self.mf:
            self.fusions = nn.ModuleList([
            self.base_fusion(features, branch_layers[l])
            for l in range(1, self.depth)])

    def forward(self, x): # (B, 1, features, latent_dim)
        h_all = []
        for i in range(self.features):
            h = self.branch_multi[i][0](x[:, 0, i]) # type: ignore
            h_all.append(h)
        h_all = torch.stack(h_all, dim=1) 

        for l in range(1, self.depth):
            if self.features > 1 and self.mf:
                h_all = self.fusions[l - 1](h_all)
            new_h_all = []
            for i in range(self.features):
                new_h_all.append(self.branch_multi[i][l](h_all[:, i])) # type: ignore
            h_all = torch.stack(new_h_all, dim=1)
        return h_all 


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
