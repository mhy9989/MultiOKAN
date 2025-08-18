import torch
from torch import nn
from .layers import get_kan


class Encoder_KAN_fusion(nn.Module):
    """fusion Encoder of KAN AE"""
    def __init__(self, H:int, W:int, AE_layers:list, kan_type = "base",
                 norm =True, actfun=nn.SiLU(), kan_config={}):
        super(Encoder_KAN_fusion, self).__init__()
        self.encoder = Encoder_KAN(H, W, AE_layers, kan_type, 
                                   norm, actfun, kan_config)

    def forward(self, x): # (B, 1, F, H, W)
        B, _, F, H, W  = x.shape
        x = x.view(B * F, H, W) 
        x = self.encoder(x) # (B*F, latent_dim)
        x = x.view(B, F, -1) 
        return x    # (B, F, latent_dim)


class Decoder_KAN_fusion(nn.Module):
    """fusion Decoder of KAN AE"""
    def __init__(self, H:int, W:int, AE_layers:list, kan_type = "base",
                 tanh=False, norm =True, actfun=nn.SiLU(), kan_config={}):
        super(Decoder_KAN_fusion, self).__init__()
        self.H, self.W = H, W
        self.decoder = Decoder_KAN(H, W, AE_layers, kan_type, 
                                   norm, actfun, kan_config)
        if tanh:
            self.decoder.append(nn.Tanh())

    def forward(self, x): # (B, F, latent_dim)
        B, F, L  = x.shape
        x = x.view(B * F, L) 
        x = self.decoder(x) # (B*F, latent_dim)
        x = x.view(B, 1, F, self.H, self.W) 
        return x    # (B, 1, F, H, W)


class Encoder_KAN_multi(nn.Module):
    """Encoder_KAN_multi"""
    def __init__(self, F:int, H:int, W:int, AE_layers:list, kan_type = "base",
                  norm =True, actfun=nn.SiLU(), kan_config={}):
        super(Encoder_KAN_multi, self).__init__()
        self.L = AE_layers[-1]
        self.encoder_multi = nn.ModuleList()
        for _ in range(F):
            self.encoder_multi.append(Encoder_KAN(H, W, AE_layers, kan_type, 
                                                  norm, actfun, kan_config))

    def forward(self, x): # (B, 1, F, H, W)
        B, _, F, H, W  = x.shape
        x = x.view(B, F, H, W) 
        output = torch.zeros(B, F, self.L, device=x.device)
        for i in range(F):
            output[:, i] = self.encoder_multi[i](x[:, i])
        return output    # (B, F, latent_dim)


class Decoder_KAN_multi(nn.Module):
    """Decoder_linear_multi"""
    def __init__(self, F:int, H:int, W:int, AE_layers:list, kan_type = "base",
                 tanh=False, norm =True, actfun=nn.SiLU(), kan_config={}):
        super(Decoder_KAN_multi, self).__init__()
        self.H, self.W = H, W
        self.decoder_multi = nn.ModuleList()
        for _ in range(F):
            decoder_layers = nn.Sequential(Decoder_KAN(H, W, AE_layers, kan_type,
                                                  norm, actfun, kan_config))
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


class Encoder_KAN(nn.Module):
    """Encoder of KAN AE"""

    def __init__(self, H:int, W:int, AE_layers:list, kan_type = "base",
                 norm =True, actfun = nn.SiLU(), kan_config={}):
        super(Encoder_KAN, self).__init__()
        en_layer = [H*W] + AE_layers
        kan_config["norm"] = norm
        self.en_kan = nn.Sequential(nn.Flatten(),
                                    get_kan(kan_type, en_layer,actfun, kan_config))

    def forward(self, x): # (B, H, W)
        x = self.en_kan(x)
        return x    # (B, latent_dim)


class Decoder_KAN(nn.Module):
    """Decoder of KAN AE"""

    def __init__(self, H:int, W:int, AE_layers:list, kan_type = "base",
                norm =True, actfun = nn.SiLU(),kan_config={}):
        super(Decoder_KAN, self).__init__()
        de_layer = AE_layers[::-1] + [H*W]
        kan_config["norm"] = norm
        if norm:
            self.de_kan = nn.Sequential(nn.LayerNorm(de_layer[0]))
        else:
            self.de_kan = nn.Sequential()
        self.de_kan.append(get_kan(kan_type, de_layer, actfun, kan_config))
        self.de_kan.append(nn.Unflatten(1, (H, W)))

    def forward(self, x): # (B, latent_dim)
        x = self.de_kan(x)
        return x    # (B, H, W)


