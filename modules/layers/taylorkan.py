# Based on this: https://github.com/Muyuzhierchengse/TaylorKAN/blob/main/TaylorKAN.ipynb
# Based on this: https://github.com/lgy112112/ikan/blob/main/ikan/TaylorKAN.py
import torch
import torch.nn.functional as F
import math

class TaylorKANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        order=3,
        scale_base=1.0,
        scale_taylor=1.0,
        base_activation=torch.nn.SiLU(),
        use_bias=True,
    ):
        super(TaylorKANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features 
        self.order = order 
        self.scale_base = scale_base
        self.scale_taylor = scale_taylor 
        self.base_activation = base_activation
        self.use_bias = use_bias 

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))

        self.taylor_coeffs = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, order)
        )

        if self.use_bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(
            self.base_weight, a=math.sqrt(5) * self.scale_base
        )

        with torch.no_grad():
            std = self.scale_taylor / (self.in_features * math.sqrt(self.order))
            self.taylor_coeffs.normal_(mean=0.0, std=std)

        if self.use_bias:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.base_weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def taylor_series(self, x: torch.Tensor):
        batch_size = x.size(0)

        x_expanded = x.unsqueeze(1).unsqueeze(-1)  # (batch_size, 1, in_features, 1)

        powers = torch.arange(self.order, device=x.device).view(1, 1, 1, -1)  # (1, 1, 1, order)
        x_powers = x_expanded ** powers  # (batch_size, 1, in_features, order)

        taylor_coeffs_expanded = self.taylor_coeffs.unsqueeze(0)  # (1, out_features, in_features, order)

        taylor_terms = x_powers * taylor_coeffs_expanded  # (batch_size, out_features, in_features, order)

        taylor_output = taylor_terms.sum(dim=3).sum(dim=2)  # (batch_size, out_features)

        return taylor_output

    def forward(self, x: torch.Tensor):
        original_shape = x.shape

        x = x.view(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)

        taylor_output = self.taylor_series(x)

        output = base_output + taylor_output

        if self.use_bias:
            output += self.bias

        output = output.view(*original_shape[:-1], self.out_features)

        return output

    def regularization_loss(self, regularize_coeffs=1.0):
        coeffs_l2 = self.taylor_coeffs.pow(2).mean()
        return regularize_coeffs * coeffs_l2


class TaylorKAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        order=3,
        scale_base=1.0,
        scale_taylor=1.0,
        base_activation=torch.nn.SiLU(),
        use_bias=True,
    ):
        super(TaylorKAN, self).__init__()

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                TaylorKANLinear(
                    in_features,
                    out_features,
                    order=order,
                    scale_base=scale_base,
                    scale_taylor=scale_taylor,
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
