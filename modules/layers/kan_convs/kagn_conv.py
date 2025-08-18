# Based on this: https://github.com/Khochawongwat/GRAMKAN/blob/main/model.py
# Based on this: https://github.com/IvanDrokin/torch-conv-kan/blob/main/kan_convs/kagn_conv.py
from functools import lru_cache
import torch
import torch.nn as nn
from torch.nn.functional import conv3d, conv2d, conv1d
import math

class KAGNConvNDLayer(nn.Module):
    def __init__(self, conv_class, conv_w_fun, input_dim, output_dim, degree, N, kernel_size,base_activation=nn.SiLU(),
                 groups=1, padding=0, stride=1, dilation=1, dropout: float = 0.0, ndim: int = 2., init="km",
                 **norm_kwargs):
        super(KAGNConvNDLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.base_activation = base_activation
        self.conv_w_fun = conv_w_fun
        self.ndim = ndim
        self.dropout = None
        self.norm_kwargs = norm_kwargs
        self.p_dropout = dropout
        self.init = init
        self.N = N
        if dropout > 0:
            if ndim == 1:
                self.dropout = nn.Dropout1d(p=dropout)
            if ndim == 2:
                self.dropout = nn.Dropout2d(p=dropout)
            if ndim == 3:
                self.dropout = nn.Dropout3d(p=dropout)

        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if input_dim % groups != 0:
            raise ValueError('input_dim must be divisible by groups')
        if output_dim % groups != 0:
            raise ValueError('output_dim must be divisible by groups')

        self.base_conv = nn.ModuleList([conv_class(input_dim // groups,
                                                   output_dim // groups,
                                                   kernel_size,
                                                   stride,
                                                   padding,
                                                   dilation,
                                                   groups=1,
                                                   bias=False) for _ in range(groups)])

        poly_shape = (groups, output_dim // groups, (input_dim // groups) * (degree + 1)) + tuple(
            kernel_size for _ in range(ndim))
        
        self.poly_weights = nn.Parameter(torch.zeros(*poly_shape))

        self.beta_weights = nn.Parameter(torch.ones(degree + 1, dtype=torch.float32))
        nn.init.normal_(
            self.beta_weights,
            mean=0.0,
            std=1.0 / ((kernel_size ** ndim) * self.inputdim * (self.degree + 1.0)),
            )

        if self.init == "km":
            for conv_layer in self.base_conv:
                nn.init.kaiming_uniform_(conv_layer.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.poly_weights, a=math.sqrt(5))
        elif self.init == "xv":
            for conv_layer in self.base_conv:
                nn.init.xavier_uniform_(conv_layer.weight)
            nn.init.xavier_uniform_(self.poly_weights)
        else:
            raise ValueError(f"Unknown init_type of conv: {self.init}")


    def beta(self, n, m):
        return (
            (m**2-n**2) * n**2 / m**2 / (4.0 * n**2 - 1.0)
        ) * self.beta_weights[n]


    @lru_cache(maxsize=128)  # Cache to avoid recomputation of Gram polynomials
    def gram_poly(self, x, degree):
        p0 = x.new_ones(x.size())

        if degree == 0:
            return p0.unsqueeze(-1)

        p1 = x
        grams_basis = [p0, p1]

        for i in range(2, degree + 1):
            p2 = x * p1 - self.beta(i - 1, self.N)  * p0
            grams_basis.append(p2)
            p0, p1 = p1, p2

        return torch.concatenate(grams_basis, dim=1)

    def forward_kag(self, x, group_index):
        # Apply base activation to input and then linear transform with base weights
        basis = self.base_conv[group_index](self.base_activation(x))

        # Normalize x to the range [-1, 1] for stable Legendre polynomial computation
        x = torch.tanh(x).contiguous()

        if self.dropout is not None:
            x = self.dropout(x)

        grams_basis = self.gram_poly(x, self.degree)

        y = self.conv_w_fun(grams_basis, self.poly_weights[group_index],
                            stride=self.stride, dilation=self.dilation,
                            padding=self.padding, groups=1)

        y = y + basis

        return y

    def forward(self, x):

        split_x = torch.split(x, self.inputdim // self.groups, dim=1)
        output = []
        for group_ind, _x in enumerate(split_x):
            y = self.forward_kag(_x, group_ind)
            output.append(y.clone())
        y = torch.cat(output, dim=1)
        return y


class KAGNConv3DLayer(KAGNConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, degree=3, N=8, groups=1, padding=0, stride=1, dilation=1,
                 dropout: float = 0.0, base_activation=nn.SiLU(), init_c="km",**norm_kwargs):
        super(KAGNConv3DLayer, self).__init__(nn.Conv3d, conv3d,
                                              input_dim, output_dim,
                                              degree, N, kernel_size,base_activation=base_activation,
                                              groups=groups, padding=padding, stride=stride, dilation=dilation,
                                              ndim=3, dropout=dropout, init=init_c, **norm_kwargs)


class KAGNConv2DLayer(KAGNConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, degree=3, N=8, groups=1, padding=0, stride=1, dilation=1,
                 dropout: float = 0.0, base_activation=nn.SiLU(), init_c="km", **norm_kwargs):
        super(KAGNConv2DLayer, self).__init__(nn.Conv2d, conv2d,
                                              input_dim, output_dim,
                                              degree, N, kernel_size,base_activation=base_activation,
                                              groups=groups, padding=padding, stride=stride, dilation=dilation,
                                              ndim=2, dropout=dropout, init=init_c,**norm_kwargs)


class KAGNConv1DLayer(KAGNConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, degree=3, N=8, groups=1, padding=0, stride=1, dilation=1,
                 dropout: float = 0.0, base_activation=nn.SiLU(), init_c="km", **norm_kwargs):
        super(KAGNConv1DLayer, self).__init__(nn.Conv1d, conv1d,
                                              input_dim, output_dim,
                                              degree, N, kernel_size,base_activation=base_activation,
                                              groups=groups, padding=padding, stride=stride, dilation=dilation,
                                              ndim=1, dropout=dropout, init=init_c, **norm_kwargs)
