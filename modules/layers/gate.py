import torch.nn as nn
import torch

class GatedFusion(nn.Module):
    def __init__(self, num_branches: int, dim: int, bottleneck: int = 32, gate_bias_init: float = -2.0):
        super().__init__()
        self.N = num_branches
        self.L = dim
        self.bn = bottleneck

        # one shared gate MLP for all (b,k) pairs to keep params small
        in_dim = 4 * dim  # [hb, hk, hb-hk, hb*hk]
        self.gate_mlp = nn.Sequential(
            nn.Linear(in_dim, bottleneck),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, dim),
        )
        # initialize last bias so sigmoid starts near 0 (like your gate_init=-2)
        nn.init.constant_(self.gate_mlp[-1].bias, gate_bias_init)

    def forward(self, x: torch.Tensor):
        B, N, L = x.shape
        fused = []

        for b in range(N):
            hb = x[:, b]                # [B, L]
            inject = torch.zeros_like(hb) # [B, L]

            for k in range(N):
                if k == b:
                    continue
                hk = x[:, k]  # [B, L]

                feat = torch.cat([hb, hk, hb - hk, hb * hk], dim=-1)  # [B, 4L]
                gate = torch.sigmoid(self.gate_mlp(feat))             # [B, L]
                inject = inject + gate * hk
            fused.append(hb + inject / (N - 1))
        return torch.stack(fused, dim=1)


class CrossStitchUnit(nn.Module):
    def __init__(self, num_branches: int, input_dim: int):
        super(CrossStitchUnit, self).__init__()
        self.num_branches = num_branches
        self.input_dim = input_dim
        W_init = torch.eye(num_branches).unsqueeze(-1).repeat(1, 1, input_dim)  # shape (N, N, L)
        self.weight = nn.Parameter(W_init) 
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.einsum('ijk, bjk -> bik', self.weight, x)
        return out


class DynamicFusion(nn.Module):
    def __init__(self, num_branches: int, input_dim: int, hidden_dim: int = 16):
        super(DynamicFusion, self).__init__()
        self.num_branches = num_branches
        self.input_dim = input_dim
        # [N -> hidden_dim -> N*N]
        self.mlp = nn.Sequential(
            nn.Linear(num_branches, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_branches * num_branches)
        )
        with torch.no_grad():
            identity_matrix = torch.eye(num_branches).reshape(-1)  # shape (N*N,)
            if self.mlp[-1].bias is not None:
                self.mlp[-1].bias.copy_(identity_matrix)

    def forward(self, x):
        """
        x: Tensor of shape (B, N, L)
        Returns:
            Tensor of shape (B, N, L) with dynamically fused features.
        """
        B, N, L = x.shape
        # Prepare input for MLP by organizing features per channel.
        x_by_feature = x.permute(0, 2, 1).reshape(B * L, N)   # shape (B*L, N)
        w_flat = self.mlp(x_by_feature)                      # shape (B*L, N*N)
        w = w_flat.view(B, L, N, N)                          # reshape to (B, L, N, N)
        # Permute w to (B, N_out, N_in, L) = (B, N, N, L) for broadcasting in einsum:
        w = w.permute(0, 2, 3, 1)                            # w[b, i, j, l] = weight from branch j to i for feature l of sample b
        # Compute fused output: out[b, i, l] = sum_j ( w[b, i, j, l] * x[b, j, l] ).
        out = torch.einsum('bjl, bijl -> bil', x, w)
        return out
