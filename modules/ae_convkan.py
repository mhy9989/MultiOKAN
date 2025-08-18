from torch import nn
import torch
from .layers import ConvKANSC, sampling_generator_kan, get_kan

class Encoder_ConvKAN_fusion(nn.Module):
    """fusion Encoder of Conv AE"""
    def __init__(self, H_m:int, W_m:int, layer_type:str, conv_type:str, AE_channels:list, kernel_size=3, 
                 norm="Batch", stride=1, act=nn.SiLU(), mid_conv=True, latent_dim=64, kan_config={}):
        super(Encoder_ConvKAN_fusion, self).__init__()
        
        self.encoder = nn.Sequential(Encoder_ConvKAN(conv_type, AE_channels, kernel_size, stride, 
                                                     norm, act, mid_conv, kan_config),
                                     nn.Flatten())
        
        if layer_type=="linear":
            self.encoder.append(act)
            self.encoder.append(nn.Linear(H_m * W_m * AE_channels[-1], latent_dim))
        else:
            self.encoder.append(get_kan(layer_type, [H_m * W_m * AE_channels[-1], latent_dim],
                                        act, kan_config))

    def forward(self, x): # (B, 1, F, H, W)
        B, _, F, H, W  = x.shape
        x = x.view(B * F, 1, H, W) 
        x = self.encoder(x) # (B*F, latent_dim)
        x = x.view(B, F, -1) 
        return x    # (B, F, latent_dim)


class Decoder_ConvKAN_fusion(nn.Module):
    """fusion Decoder of Conv AE"""
    def __init__(self, H_m:int, W_m:int, layer_type:str, conv_type:str, tanh:bool, AE_channels:list, kernel_size=3,
                 norm="Batch", stride=1, act=nn.SiLU(), mid_conv=True, latent_dim=64, kan_config={}):
        super(Decoder_ConvKAN_fusion, self).__init__()
        
        if layer_type=="linear":
            self.decoder = nn.Sequential(nn.Linear(latent_dim, H_m * W_m * AE_channels[-1]))
        else:
            self.decoder = nn.Sequential(get_kan(layer_type, [latent_dim, H_m * W_m * AE_channels[-1]],  
                                                act, kan_config))
            
        self.decoder.append(nn.Unflatten(1, (AE_channels[-1], H_m, W_m)))
        self.decoder.append(Decoder_ConvKAN(conv_type, AE_channels, kernel_size,stride, 
                                            norm, act, mid_conv, kan_config))
        if tanh:
            self.decoder.append(nn.Tanh())

    def forward(self, x): # (B, F, latent_dim)
        B, F, L  = x.shape
        x = x.view(B * F, L) 
        x = self.decoder(x) # (B*F, 1, H, W)
        _, _, H, W = x.shape
        x = x.view(B, 1, F, H, W) 
        return x    # (B, 1, F, H, W)


class Encoder_ConvKAN_multi(nn.Module):
    """Encoder_ConvKAN_multi"""
    def __init__(self, F:int, H_m:int, W_m:int, layer_type:str, conv_type:str, AE_channels:list, kernel_size=3,
                 norm="Batch",  stride=1, act=nn.SiLU(), mid_conv=True, latent_dim=64, kan_config={}):
        super(Encoder_ConvKAN_multi, self).__init__()
        self.encoder_multi = nn.ModuleList()
        self.L = latent_dim
        for _ in range(F):
            encoder_layer = nn.Sequential(
                Encoder_ConvKAN(conv_type, AE_channels, kernel_size,stride, norm,act,mid_conv,kan_config),
                nn.Flatten())
            if layer_type=="linear":
                encoder_layer.append(act)
                encoder_layer.append(nn.Linear(H_m * W_m * AE_channels[-1], latent_dim))
            else:
                encoder_layer.append(get_kan(layer_type, [H_m * W_m * AE_channels[-1], latent_dim], 
                                            act, kan_config))
            self.encoder_multi.append(encoder_layer)

    def forward(self, x): # (B, 1, F, H, W)
        B, _, F, _, _  = x.shape
        output = torch.zeros(B, F, self.L, device=x.device)
        for i in range(F):
            output[:, i] = self.encoder_multi[i](x[:, :, i])# (B, latent_dim)
        return output    # (B, F, latent_dim)


class Decoder_ConvKAN_multi(nn.Module):
    """Decoder_ConvKAN_multi"""
    def __init__(self, F:int, H:int, W:int, H_m:int, W_m:int, layer_type:str, conv_type:str, tanh:bool, 
                 AE_channels:list, kernel_size=3, norm="Batch", stride=1, act=nn.SiLU(), mid_conv=True, 
                 latent_dim=64, kan_config={}):
        super(Decoder_ConvKAN_multi, self).__init__()
        self.H, self.W = H, W
        self.decoder_multi = nn.ModuleList()
        for _ in range(F):
            if layer_type=="linear":
                decoder_layer = nn.Sequential(nn.Linear(latent_dim, H_m * W_m * AE_channels[-1]))
            else:
                decoder_layer = nn.Sequential(get_kan(layer_type, [latent_dim, H_m * W_m * AE_channels[-1]],  
                                                        act,kan_config))
                
            decoder_layer.append(nn.Unflatten(1, (AE_channels[-1], H_m, W_m)))
            decoder_layer.append(Decoder_ConvKAN(conv_type, AE_channels, kernel_size,stride, 
                                                 norm, act, mid_conv, kan_config))
            self.decoder_multi.append(decoder_layer)
            if tanh:
                decoder_layer.append(nn.Tanh())
            self.decoder_multi.append(decoder_layer)


    def forward(self, x): # (B, F, latent_dim)
        B, F, _  = x.shape
        output = torch.zeros(B, 1, F, self.H, self.W, device=x.device)
        for i in range(F):
            output[:, :, i] = self.decoder_multi[i](x[:, i])# (B, 1, H, W)
        return output    # (B, 1, F, H, W)


class Encoder_ConvKAN(nn.Module):
    """Encoder of ConvKAN AE"""

    def __init__(self, conv_type, AE_channels, kernel_size, stride, norm, act, mid_conv=True,
                 kan_config={}):
        N_E = len(AE_channels)
        samplings = sampling_generator_kan(N_E, mid_conv=mid_conv)
        super(Encoder_ConvKAN, self).__init__()
        self.encoder = nn.Sequential(
              ConvKANSC(1, AE_channels[0], kernel_size, stride, downsampling=samplings[0],
                     norm=norm, act_fun=act, conv_type=conv_type, kan_config=kan_config),
            *[ConvKANSC(AE_channels[i], AE_channels[i+1], kernel_size, stride, downsampling=samplings[i+1],
                     norm=norm, act_fun=act, conv_type=conv_type, kan_config=kan_config) for i in range(N_E-1)]
        )

    def forward(self, x):  
        latent = self.encoder(x)
        return latent


class Decoder_ConvKAN(nn.Module):
    """DEcoder of ConvKAN AE"""

    def __init__(self, conv_type, AE_channels, kernel_size, stride, norm, act, mid_conv=True,
                 kan_config={}):
        N_D = len(AE_channels)
        samplings = sampling_generator_kan(N_D, reverse=True, mid_conv=mid_conv)
        super(Decoder_ConvKAN, self).__init__()
        self.decoder = nn.Sequential(
            *[ConvKANSC(AE_channels[-i], AE_channels[-i-1], kernel_size, stride, upsampling=samplings[i-1],
                     norm=norm, act_fun=act, conv_type=conv_type, kan_config=kan_config) for i in range(1, N_D)],
              ConvKANSC(AE_channels[0], 1, kernel_size, stride, upsampling=samplings[-1],
                     norm=False, act_fun=act, conv_type=conv_type, kan_config=kan_config)
        )

    def forward(self, latent):
        Y = self.decoder(latent)
        return Y
