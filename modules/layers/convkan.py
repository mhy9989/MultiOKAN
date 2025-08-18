from torch import nn
from .kan_convs import KANConv2DLayer, KAGNConv2DLayer, BottleNeckKAGNConv2DLayer

class BasicConvKAN2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 expend=2,
                 padding=0,
                 dilation=1,
                 upsampling=False,
                 norm=False,
                 act_fun=nn.SiLU(),
                 conv_type="kagn",
                 kan_config={}):
        super(BasicConvKAN2d, self).__init__()
        self.norm = norm
        self.act = act_fun
        if conv_type=="bkagn":
            self.base_conv = BottleNeckKAGNConv2DLayer 
        elif conv_type=="kagn":
            self.base_conv = KAGNConv2DLayer
        elif conv_type=="base":
            self.base_conv = KANConv2DLayer
        else:
            raise ValueError(f"Unknown conv_type type: {conv_type}")
        
        if upsampling is True:
            self.conv = nn.Sequential(*[
                self.base_conv(in_channels, out_channels*(expend**2), kernel_size=kernel_size,
                          stride=1, padding=padding, dilation=dilation, base_activation = act_fun,
                          **kan_config),
                nn.PixelShuffle(expend)
            ])
        else:
            self.conv = nn.Sequential(self.base_conv(
                in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation, base_activation = act_fun,
                **kan_config))

        if norm == "Batch":
            self.norm_fun = nn.BatchNorm2d(out_channels)
        elif norm == "Group":
            if out_channels % 2 == 0:
                self.norm_fun = nn.GroupNorm(2, out_channels)
            else:
                self.norm_fun = nn.GroupNorm(1, out_channels)

    def forward(self, x):
        y = self.conv(x)
        if self.norm:
            y = self.norm_fun(y)
        return y


class ConvKANSC(nn.Module):

    def __init__(self,
                 C_in,
                 C_out,
                 kernel_size=3,
                 stride=1,
                 downsampling=False,
                 upsampling=False,
                 norm="Batch",
                 act_fun = nn.SiLU(),
                 conv_type="kagn",
                 kan_config={}):
        super(ConvKANSC, self).__init__()

        stride_n = stride if downsampling is True else 1
        padding = (kernel_size - stride_n + 1) // 2

        self.conv = BasicConvKAN2d(C_in, C_out, kernel_size=kernel_size, stride=stride_n,
                                   expend=stride,upsampling=upsampling, padding=padding,
                                   norm=norm, act_fun=act_fun, conv_type=conv_type,
                                   kan_config=kan_config)

    def forward(self, x):
        y = self.conv(x)
        return y