import torch
from torch import nn
from .layers import get_kan, get_kan_layer, GatedFusion, CrossStitchUnit, DynamicFusion

class Branch_multi_KAN(nn.Module):
    """Branch_multi_KAN"""
    def __init__(self, features:int, p:int, latent_dim:int, layers:list, tf=True, mf=True,
                 branch_type="gram",norm = True, actfun=nn.SiLU(), kan_config={}):
        super(Branch_multi_KAN, self).__init__()
        self.features = features
        self.p = p
        self.latent_dim = latent_dim
        kan_config["norm"] = norm
        self.mf=mf
        self.base_fusion=DynamicFusion
        self.f_dim = features if tf else 1
        last_dim = self.f_dim * latent_dim * p

        branch_layers = [latent_dim] + layers + [last_dim]
        self.branch_multi = nn.ModuleList()
        kan_layer = get_kan_layer(branch_type)
        self.depth = len(branch_layers) -1
        
        for _ in range(features):
            branch=nn.ModuleList()
            for i, (in_features, out_features) in enumerate(zip(branch_layers, branch_layers[1:])): 
                layer = [kan_layer(
                    in_features,
                    out_features,
                    base_activation=actfun,
                    **kan_config
                    )]
                if i < self.depth - 1 and norm == True:
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
        return h_all    # (B, features, features/1, latent_dim, p)

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

