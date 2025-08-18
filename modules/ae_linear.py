import torch
from torch import nn


class Encoder_Linear_fusion(nn.Module):
    """fusion Encoder of Linear AE"""
    def __init__(self, H:int, W:int, AE_layers:list, norm =True, actfun=nn.SiLU()):
        super(Encoder_Linear_fusion, self).__init__()
        self.encoder = Encoder_Linear(H, W, AE_layers, norm, actfun)

    def forward(self, x): # (B, 1, F, H, W)
        B, _, F, H, W  = x.shape
        x = x.view(B * F, H, W) 
        x = self.encoder(x) # (B*F, latent_dim)
        x = x.view(B, F, -1) 
        return x    # (B, F, latent_dim)


class Decoder_Linear_fusion(nn.Module):
    """fusion Decoder of Linear AE"""
    def __init__(self, H:int, W:int, AE_layers:list, tanh=False, norm =True, actfun=nn.SiLU()):
        super(Decoder_Linear_fusion, self).__init__()
        self.H, self.W = H, W
        self.decoder = Decoder_Linear(H, W, AE_layers, norm, actfun)
        if tanh:
            self.decoder.append(nn.Tanh())

    def forward(self, x): # (B, F, latent_dim)
        B, F, L  = x.shape
        x = x.view(B * F, L) 
        x = self.decoder(x) # (B*F, latent_dim)
        x = x.view(B, 1, F, self.H, self.W) 
        return x    # (B, 1, F, H, W)


class Encoder_Linear_multi(nn.Module):
    """Encoder_Linear_multi"""
    def __init__(self, F:int, H:int, W:int, AE_layers:list, norm =True, actfun=nn.SiLU()):
        super(Encoder_Linear_multi, self).__init__()
        self.L = AE_layers[-1]
        self.encoder_multi = nn.ModuleList()
        for _ in range(F):
            self.encoder_multi.append(Encoder_Linear(H, W, AE_layers, norm, actfun))

    def forward(self, x): # (B, 1, F, H, W)
        B, _, F, H, W  = x.shape
        x = x.view(B, F, H, W) 
        output = torch.zeros(B, F, self.L, device=x.device)
        for i in range(F):
            output[:, i] = self.encoder_multi[i](x[:, i])
        return output    # (B, F, latent_dim)


class Decoder_Linear_multi(nn.Module):
    """Decoder_linear_multi"""
    def __init__(self, F:int, H:int, W:int, AE_layers:list, tanh=False, norm =True, actfun=nn.SiLU()):
        super(Decoder_Linear_multi, self).__init__()
        self.H, self.W = H, W
        self.decoder_multi = nn.ModuleList()
        for _ in range(F):
            decoder_layers = nn.Sequential(Decoder_Linear(H, W, AE_layers, norm, actfun))
            if tanh:
                decoder_layers.append(nn.Tanh())
            self.decoder_multi.append(decoder_layers)

    def forward(self, x): # (B, F, latent_dim)
        B, F, _  = x.shape
        output = torch.zeros(B, F, self.H, self.W, device=x.device)
        for i in range(F):
            output[:, i] = self.decoder_multi[i](x[:, i])
        output = output.view(B, 1, F, self.H, self.W) 
        return output    # (B, 1, F, H, W)


class Encoder_Linear(nn.Module):
    """Encoder of Linear AE"""

    def __init__(self, H:int, W:int, AE_layers:list, norm =True, actfun=nn.SiLU()):
        super(Encoder_Linear, self).__init__()
        en_layer=[H * W]+ AE_layers
        N_E = len(en_layer)
        self.encoder = nn.Sequential(nn.Flatten(),nn.Linear(en_layer[0], en_layer[1]))
        for i in range(1, N_E-1):
            self.encoder.append(actfun)
            if norm:
                self.encoder.append(nn.LayerNorm(en_layer[i]))
            self.encoder.append(nn.Linear(en_layer[i], en_layer[i+1]))

    def forward(self, x): # (B, H, W)
        x = self.encoder(x)
        return x    # (B, latent_dim)


class Decoder_Linear(nn.Module):
    """Decoder of Linear AE"""

    def __init__(self, H:int, W:int, AE_layers:list, norm =True, actfun=nn.SiLU()):
        super(Decoder_Linear, self).__init__()
        de_layer = AE_layers[::-1] + [H*W]
        N_D = len(de_layer)
        self.decoder = nn.Sequential()
        for i in range(N_D-1):
            self.decoder.append(actfun)
            if norm:
                self.decoder.append(nn.LayerNorm(de_layer[i]))
            self.decoder.append(nn.Linear(de_layer[i], de_layer[i+1]))
        self.decoder.append(nn.Unflatten(1, (H, W)))

    def forward(self, x): # (B, latent_dim)
        x = self.decoder(x)
        return x    # (B, H, W)
