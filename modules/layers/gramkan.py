# Based on this: https://github.com/Khochawongwat/GRAMKAN/blob/main/model.py
from functools import lru_cache
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GRAMLayer(nn.Module):
    def __init__(self, in_channels, out_channels, degrees = 3, N = 8, base_activation=nn.SiLU(), init="xv"):
        super(GRAMLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.degrees = degrees
        self.init = init
        self.N = N

        self.act = base_activation
        self.beta_weights = nn.Parameter(torch.ones(degrees + 1, dtype=torch.float32))

        self.grams_basis_weights = nn.Parameter(
            torch.zeros(out_channels, in_channels, degrees + 1, dtype=torch.float32)
        )

        self.base_weights = nn.Parameter(
                torch.zeros(out_channels, in_channels, dtype=torch.float32)
            )

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(
                self.beta_weights,
                mean=0.0,
                std=1.0 / (self.in_channels * (self.degrees + 1.0)),
        )
        if self.init == "xv":
            nn.init.xavier_uniform_(self.grams_basis_weights)
            nn.init.xavier_uniform_(self.base_weights)
        elif self.init == "km":
            nn.init.kaiming_uniform_(self.grams_basis_weights, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.base_weights, a=math.sqrt(5))
        else:
            raise ValueError(f"Unknown init_type of linear: {self.init}")

    def beta(self, n, m):
        return (
            (m**2-n**2) * n**2 / m**2 / (4.0 * n**2 - 1.0)
        ) * self.beta_weights[n]

    @lru_cache(maxsize=128)
    def gram_poly(self, x, degree):
        p0 = x.new_ones(x.size())

        if degree == 0:
            return p0.unsqueeze(-1)

        p1 = x
        grams_basis = [p0, p1]

        for i in range(2, degree + 1):
            p2 = x * p1 - self.beta(i - 1, self.N) * p0
            grams_basis.append(p2)
            p0, p1 = p1, p2

        return torch.stack(grams_basis, dim=-1)

    def forward(self, x):
        basis = F.linear(self.act(x), self.base_weights)

        x = torch.tanh(x).contiguous()

        grams_basis = self.gram_poly(x, self.degrees)

        y = F.linear(
            grams_basis.view(x.size(0), -1),
            self.grams_basis_weights.view(self.out_channels, -1))

        y = y + basis

        y = y.view(-1, self.out_channels)
        
        return y


class GRAMKAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        degrees=3,
        base_activation=torch.nn.SiLU(),
        norm=False,
        init_l="xv",
        **kwargs
    ):
        super(GRAMKAN, self).__init__()

        self.layers = torch.nn.ModuleList()
        for i, (in_features, out_features) in enumerate(zip(layers_hidden, layers_hidden[1:])): 
            self.layers.append(
                GRAMLayer(
                    in_features,
                    out_features,
                    degrees=degrees,
                    base_activation=base_activation,
                    init=init_l
                )
            )
            if i < len(layers_hidden) - 2 and norm == True:
                self.layers.append(torch.nn.LayerNorm(out_features))
                

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x
