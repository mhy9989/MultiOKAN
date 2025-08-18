import math
from torch import nn
from timm.layers  import trunc_normal_
from core.act_funs import none_act

class BasicConv2d(nn.Module):

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
                 act_fun = nn.Mish()):
        super(BasicConv2d, self).__init__()
        self.norm = norm
        if upsampling is True:
            self.conv = nn.Sequential(*[
                nn.Conv2d(in_channels, out_channels*(expend**2), kernel_size=kernel_size,
                          stride=1, padding=padding, dilation=dilation),
                nn.PixelShuffle(expend)
            ])
        else:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation)

        if norm == "Batch":
            self.norm_fun = nn.BatchNorm2d(out_channels)
        elif norm == "Group":
            if out_channels % 2 == 0:
                self.norm_fun = nn.GroupNorm(2, out_channels)
            else:
                self.norm_fun = nn.GroupNorm(1, out_channels)
        
        self.act = act_fun if act_fun else none_act()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.conv(x)
        if self.norm =="Batch":
            y = self.norm_fun(self.act(y))
        elif self.norm =="Group":
            y = self.act(self.norm_fun(y))
        else:
            y = self.act(y)
        return y


class ConvSC(nn.Module):

    def __init__(self,
                 C_in,
                 C_out,
                 kernel_size=3,
                 stride=1,
                 downsampling=False,
                 upsampling=False,
                 norm="Batch",
                 act_fun = nn.Mish(),
                 **kwargs):
        super(ConvSC, self).__init__()

        stride_n = stride if downsampling is True else 1
        padding = (kernel_size - stride_n + 1) // 2

        self.conv = BasicConv2d(C_in, C_out, kernel_size=kernel_size, stride=stride_n,
                                   expend=stride,upsampling=upsampling, padding=padding,
                                   norm=norm, act_fun=act_fun)

    def forward(self, x):
        y = self.conv(x)
        return y
    

def sampling_generator(N, reverse=False, mid_conv=True):
    samplings_batch = [True, False] if mid_conv else [True, True]
    samplings = samplings_batch * math.ceil(N / 2)
    if reverse: return list(reversed(samplings[:N]))
    else: return samplings[:N]
