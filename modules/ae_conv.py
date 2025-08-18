from torch import nn
import torch
from .layers import ConvSC, sampling_generator


class Encoder_Conv_fusion(nn.Module):
    """fusion Encoder of Conv AE"""
    def __init__(self, H_m:int, W_m:int, AE_channels:list, kernel_size=3,
                 norm="Batch", stride=1,act=nn.SiLU(), mid_conv=True, latent_dim=64):
        super(Encoder_Conv_fusion, self).__init__()
        self.encoder = nn.Sequential(Encoder_Conv(AE_channels, kernel_size,stride,norm,act,mid_conv),
                                     nn.Flatten(),
                                     nn.Linear(H_m * W_m * AE_channels[-1], latent_dim))

    def forward(self, x): # (B, 1, F, H, W)
        B, _, F, H, W  = x.shape
        x = x.view(B * F, 1, H, W) 
        x = self.encoder(x) # (B*F, latent_dim)
        x = x.view(B, F, -1) 
        return x    # (B, F, latent_dim)


class Decoder_Conv_fusion(nn.Module):
    """fusion Decoder of Conv AE"""
    def __init__(self, H_m:int, W_m:int, AE_channels:list, tanh:bool, kernel_size=3,
                 norm="Batch", stride=1,act=nn.SiLU(), mid_conv=True, latent_dim=64):
        super(Decoder_Conv_fusion, self).__init__()
        self.decoder = nn.Sequential(nn.Linear(latent_dim, H_m * W_m * AE_channels[-1]),
                                nn.Unflatten(1, (AE_channels[-1], H_m, W_m)),
                                Decoder_Conv(AE_channels, kernel_size,stride,norm,act,mid_conv))
        if tanh:
            self.decoder.append(nn.Tanh())

    def forward(self, x): # (B, F, latent_dim)
        B, F, L  = x.shape
        x = x.view(B * F, L) 
        x = self.decoder(x) # (B*F, 1, H, W)
        _, _, H, W = x.shape
        x = x.view(B, 1, F, H, W) 
        return x    # (B, 1, F, H, W)


class Encoder_Conv_multi(nn.Module):
    """Encoder_Conv_multi"""
    def __init__(self, F:int, H_m:int, W_m:int, AE_channels:list, kernel_size=3,
                 norm="Batch", stride=1,act=nn.SiLU(), mid_conv=True, latent_dim=64):
        super(Encoder_Conv_multi, self).__init__()
        self.encoder_multi = nn.ModuleList()
        self.L = latent_dim
        for _ in range(F):
            self.encoder_multi.append(nn.Sequential(
                Encoder_Conv(AE_channels, kernel_size,stride,norm,act,mid_conv),
                nn.Flatten(),
                nn.Linear(H_m * W_m * AE_channels[-1], latent_dim)))

    def forward(self, x): # (B, 1, F, H, W)
        B, _, F, _, _  = x.shape
        output = torch.zeros(B, F, self.L, device=x.device)
        for i in range(F):
            output[:, i] = self.encoder_multi[i](x[:, :, i])# (B, latent_dim)
        return output    # (B, F, latent_dim)


class Decoder_Conv_multi(nn.Module):
    """Decoder_Conv_multi"""
    def __init__(self, F:int, H:int, W:int, H_m:int, W_m:int, AE_channels:list, tanh:bool, 
                 kernel_size=3, norm="Batch", stride=1,act=nn.SiLU(), mid_conv=True, latent_dim=64):
        super(Decoder_Conv_multi, self).__init__()
        self.H, self.W = H, W
        self.decoder_multi = nn.ModuleList()
        for _ in range(F):
            decoder_layer=nn.Sequential(
                nn.Linear(latent_dim, H_m * W_m * AE_channels[-1]),
                nn.Unflatten(1, (AE_channels[-1], H_m, W_m)),
                Decoder_Conv(AE_channels, kernel_size,stride,norm,act,mid_conv))
            if tanh:
                decoder_layer.append(nn.Tanh())
            self.decoder_multi.append(decoder_layer)

    def forward(self, x): # (B, F, latent_dim)
        B, F, _  = x.shape
        output = torch.zeros(B, 1, F, self.H, self.W, device=x.device)
        for i in range(F):
            output[:, :, i] = self.decoder_multi[i](x[:, i])# (B, 1, H, W)
        return output    # (B, 1, F, H, W)


class Encoder_Conv(nn.Module):
    """Encoder of Conv AE"""

    def __init__(self, AE_channels, kernel_size, stride, norm, act, mid_conv=True):
        N_E = len(AE_channels)
        samplings = sampling_generator(N_E, mid_conv=mid_conv)
        super(Encoder_Conv, self).__init__()
        self.encoder = nn.Sequential(
              ConvSC(1, AE_channels[0], kernel_size, stride,downsampling=samplings[0],
                     norm=norm, act_fun=act),
            *[ConvSC(AE_channels[i], AE_channels[i+1], kernel_size, stride,downsampling=samplings[i+1],
                     norm=norm, act_fun=act) for i in range(N_E-1)]
        )

    def forward(self, x):  
        latent = self.encoder(x)
        return latent


class Decoder_Conv(nn.Module):
    """DEcoder of Conv AE"""

    def __init__(self, AE_channels, kernel_size, stride, norm, act, mid_conv=True):
        N_D = len(AE_channels)
        samplings = sampling_generator(N_D, reverse=True, mid_conv=mid_conv)
        super(Decoder_Conv, self).__init__()
        self.decoder = nn.Sequential(act,
            *[ConvSC(AE_channels[-i], AE_channels[-i-1], kernel_size, stride,upsampling=samplings[i-1],
                     norm=norm, act_fun=act) for i in range(1, N_D)],
              ConvSC(AE_channels[0], 1, kernel_size, stride,upsampling=samplings[-1],
                     norm=False, act_fun=False)
        )

    def forward(self, latent):
        Y = self.decoder(latent)
        return Y
