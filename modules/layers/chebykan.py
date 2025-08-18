# Based on this: https://github.com/SynodicMonth/ChebyKAN/blob/main/ChebyKANLayer.py
# Based on this: https://github.com/lgy112112/ikan/blob/main/ikan/ChebyKAN.py
import torch
import torch.nn.functional as F
import math

class ChebyKANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        degree=5,
        scale_base=1.0,
        scale_cheby=1.0,
        base_activation=torch.nn.SiLU(),
        use_bias=True,
    ):
        super(ChebyKANLinear, self).__init__()
        self.in_features = in_features  
        self.out_features = out_features
        self.degree = degree 
        self.scale_base = scale_base
        self.scale_cheby = scale_cheby
        self.base_activation = base_activation
        self.use_bias = use_bias

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))

        self.cheby_coeffs = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, degree + 1)
        )

        if self.use_bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)

        self.register_buffer("cheby_orders", torch.arange(0, degree + 1).float())

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)

        with torch.no_grad():
            std = self.scale_cheby / math.sqrt(self.in_features)
            self.cheby_coeffs.uniform_(-std, std)

        if self.use_bias:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.base_weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def chebyshev_polynomials(self, x: torch.Tensor):
        x = torch.tanh(x.clamp(-1,1))

        theta = torch.acos(x)

        theta_n = theta.unsqueeze(-1) * self.cheby_orders

        T_n = torch.cos(theta_n)

        return T_n

    def forward(self, x: torch.Tensor):
        original_shape = x.shape

        x = x.view(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)

        T_n = self.chebyshev_polynomials(x)

        cheby_output = torch.einsum('bik,oik->bo', T_n, self.cheby_coeffs)

        output = base_output + cheby_output

        if self.use_bias:
            output += self.bias

        output = output.view(*original_shape[:-1], self.out_features)

        return output

    def regularization_loss(self, regularize_coeffs=1.0):
        coeffs_l2 = self.cheby_coeffs.pow(2).mean()
        return regularize_coeffs * coeffs_l2

class ChebyKAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        degree=7,
        scale_base=1.0,
        scale_cheby=1.0,
        base_activation=torch.nn.SiLU(),
        use_bias=False,
    ):
        super(ChebyKAN, self).__init__()

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                ChebyKANLinear(
                    in_features,
                    out_features,
                    degree=degree,
                    scale_base=scale_base,
                    scale_cheby=scale_cheby,
                    base_activation=base_activation,
                    use_bias=use_bias,
                )
            )

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x

    def regularization_loss(self, regularize_coeffs=1.0):
        return sum(
            layer.regularization_loss(regularize_coeffs)
            for layer in self.layers
        )
