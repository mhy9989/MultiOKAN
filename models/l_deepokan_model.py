from torch import nn
import torch
from core import act_list
from modules import *
import math

class L_DeepOKan_Model_AE_fusion(nn.Module):
    def __init__(self, input_shape, AE_layers, kan_type = "base", tanh = False,
                norm =True, actfun="silu", kan_config={}, **kwargs):
        super(L_DeepOKan_Model_AE_fusion, self).__init__()
        _, H, W = input_shape
        self.encoder = Encoder_KAN_fusion(H, W, AE_layers, kan_type, 
                                          norm, act_list[actfun.lower()], 
                                          kan_config.copy())
        self.decoder = Decoder_KAN_fusion(H, W, AE_layers, kan_type, tanh,
                                          norm, act_list[actfun.lower()], 
                                          kan_config.copy())

    def forward(self, x, **kwargs):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class L_DeepOKan_Model_AE_multi(nn.Module):
    def __init__(self, input_shape, AE_layers, kan_type = "base", tanh = False,
                 norm =True, actfun="silu", kan_config={}, **kwargs):
        super(L_DeepOKan_Model_AE_multi, self).__init__()
        F, H, W = input_shape
        self.encoder = Encoder_KAN_multi(F, H, W, AE_layers, kan_type, 
                                         norm, act_list[actfun.lower()], 
                                         kan_config.copy())
        self.decoder = Decoder_KAN_multi(F, H, W, AE_layers, kan_type, tanh,
                                         norm, act_list[actfun.lower()], 
                                         kan_config.copy())

    def forward(self, x, **kwargs):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class L_DeepOKan_Model_AE_Conv_fusion(nn.Module):
    def __init__(self, input_shape, AE_channels, latent_dim, layer_type = "linear", conv_type="kagn", tanh = False,
                 actfun="silu",norm="Batch",kernel_size=3, stride=2, mid_conv=True,kan_config={},**kwargs):
        super(L_DeepOKan_Model_AE_Conv_fusion, self).__init__()
        _, H, W = input_shape

        if mid_conv:
            samplings_size = stride **(math.ceil(len(AE_channels)/ 2))
        else:
            samplings_size = stride ** len(AE_channels)
        
        H_m, W_m = H // samplings_size, W // samplings_size
        self.encoder = Encoder_ConvKAN_fusion(H_m, W_m, layer_type, conv_type,
                                            AE_channels, kernel_size, norm, stride,
                                            act_list[actfun.lower()], mid_conv, latent_dim,
                                            kan_config.copy())
        self.decoder = Decoder_ConvKAN_fusion(H_m, W_m, layer_type, conv_type, tanh,
                                            AE_channels, kernel_size, norm, stride,
                                            act_list[actfun.lower()], mid_conv, latent_dim,
                                            kan_config.copy())

    def forward(self, x, **kwargs):
        # (B, 1, features, H, W)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class L_DeepOKan_Model_AE_Conv_multi(nn.Module):
    def __init__(self, input_shape, AE_channels, latent_dim, layer_type = "linear", conv_type="kagn", tanh = False,
                 actfun="silu",norm="Batch",kernel_size=3, stride=2, mid_conv=True,kan_config={},**kwargs):
        super(L_DeepOKan_Model_AE_Conv_multi, self).__init__()
        F, H, W = input_shape
        if mid_conv:
            samplings_size = stride **(math.ceil(len(AE_channels)/ 2))
        else:
            samplings_size = stride ** len(AE_channels)


        H_m, W_m = H // samplings_size, W // samplings_size

        self.encoder = Encoder_ConvKAN_multi(F, H_m, W_m, layer_type, conv_type,
                                            AE_channels, kernel_size, norm, stride,
                                            act_list[actfun.lower()], mid_conv, latent_dim,
                                            kan_config.copy())
        self.decoder = Decoder_ConvKAN_multi(F, H, W, H_m, W_m, layer_type, conv_type, tanh,
                                            AE_channels, kernel_size, norm,  
                                            stride, act_list[actfun.lower()], mid_conv, latent_dim,
                                            kan_config.copy())

    def forward(self, x, **kwargs):
        # (B, 1, features, H, W)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class L_DeepOKan_Model_DON_multi(nn.Module):
    def __init__(self, features, p, latent_dim, branch_layers, trunk_layers, norm = [True,False], branch_type="gram",
                 trunk_type="gram", branch_actfun="silu", trunk_actfun="silu", kan_config={}, **kwargs):
        super(L_DeepOKan_Model_DON_multi, self).__init__()
        self.m = latent_dim
        self.features = features
        self.branch = Branch_multi_KAN(features, p, latent_dim, branch_layers, branch_type, 
                                        norm[0], act_list[branch_actfun.lower()], 
                                        kan_config.copy())
        if trunk_type == "linear":
            self.trunk = Trunk(p, latent_dim, trunk_layers,
                                norm[1], act_list[trunk_actfun.lower()])
        else:
            self.trunk = Trunk_KAN(p, latent_dim, trunk_layers, trunk_type,
                                    norm[1], act_list[trunk_actfun.lower()], 
                                    kan_config.copy())

    def forward(self, x0, x1, **kwargs):
        #x0:  # (B, 1, features, latent_dim)
        #x1:  # (nt, 1)
        y_branch = self.branch(x0) # (B, features, features, latent_dim, p)
        y_trunk = self.trunk(x1)   # (nt, latent_dim, p)
        y_branch_mul = torch.prod(y_branch, dim=1) # (B, features, latent_dim, p)
        Y = torch.einsum('ijkl,pkl->ipjk', y_branch_mul, y_trunk) # (B, nt, features, latent_dim)
        return Y


class L_DeepOKan_Model_DON_Conv_multi(nn.Module):
    def __init__(self, features, p, latent_dim, trunk_layers, branch_channels,
                 kernel_size=3, norm=["Batch",False],
                 conv_type="kagn", layer_type = "gram", branch_actfun="silu",
                 trunk_type="gram", trunk_actfun="silu", kan_config={}, **kwargs):
        super(L_DeepOKan_Model_DON_Conv_multi, self).__init__()
        self.m = latent_dim
        self.features = features
        self.branch = Branch_ConvKAN_multi(features, p, latent_dim, conv_type, layer_type, 
                                           branch_channels, kernel_size, norm[0], 
                                           act_list[branch_actfun.lower()],kan_config.copy())
        if trunk_type == "linear":
            self.trunk = Trunk(p, latent_dim, trunk_layers,
                               norm[1], act_list[trunk_actfun.lower()])
        else:
            self.trunk = Trunk_KAN(p, latent_dim, trunk_layers, trunk_type, 
                                   norm[1],act_list[trunk_actfun.lower()], 
                                   kan_config.copy())
    
    def forward(self, x0, x1, **kwargs):
        #x0:  # (B, 1, features, latent_dim)
        #x1:  # (nt, 1)
        y_branch = self.branch(x0) # (B, features, features, latent_dim, p)
        y_trunk = self.trunk(x1)   # (nt, latent_dim, p)
        y_branch_mul = torch.prod(y_branch, dim=1) # (B, features, latent_dim, p)
        Y = torch.einsum('ijkl,pkl->ipjk', y_branch_mul, y_trunk) # (B, nt, features, latent_dim)
        return Y