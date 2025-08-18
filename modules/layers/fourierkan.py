# Based on this: https://github.com/GistNoesis/FourierKAN/blob/main/fftKAN.py
# Based on this: https://github.com/lgy112112/ikan/blob/main/ikan/FourierKAN.py
import torch
import torch.nn.functional as F
import math

class FourierKANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        num_frequencies=10,
        scale_base=1.0,
        scale_fourier=1.0,
        base_activation=torch.nn.SiLU(),
        use_bias=True,
        smooth_initialization=False,
    ):
        super(FourierKANLinear, self).__init__()
        self.in_features = in_features 
        self.out_features = out_features 
        self.num_frequencies = num_frequencies 
        self.scale_base = scale_base 
        self.scale_fourier = scale_fourier  
        self.base_activation = base_activation 
        self.use_bias = use_bias  
        self.smooth_initialization = smooth_initialization  

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))

        self.fourier_coeffs = torch.nn.Parameter(
            torch.Tensor(2, out_features, in_features, num_frequencies)
        )

        if self.use_bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)

        with torch.no_grad():
            if self.smooth_initialization:
                frequency_decay = (torch.arange(self.num_frequencies, device=self.fourier_coeffs.device) + 1.0) ** -2.0
            else:
                frequency_decay = torch.ones(self.num_frequencies, device=self.fourier_coeffs.device)
            
            std = self.scale_fourier / math.sqrt(self.in_features) / frequency_decay 
            std = std.view(1, 1, -1)

            self.fourier_coeffs[0].uniform_(-1, 1)
            self.fourier_coeffs[0].mul_(std)

            self.fourier_coeffs[1].uniform_(-1, 1)
            self.fourier_coeffs[1].mul_(std)


        if self.use_bias:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.base_weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor):
        original_shape = x.shape
        x = x.view(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)

        k = torch.arange(1, self.num_frequencies + 1, device=x.device).view(1, 1, -1)

        x_expanded = x.unsqueeze(-1)

        xk = x_expanded * k 

        cos_xk = torch.cos(xk)
        sin_xk = torch.sin(xk)

        cos_part = torch.einsum(
            "bif, oif->bo",
            cos_xk,
            self.fourier_coeffs[0],
        )
        sin_part = torch.einsum(
            "bif, oif->bo",
            sin_xk,
            self.fourier_coeffs[1],
        )

        fourier_output = cos_part + sin_part

        output = base_output + fourier_output

        if self.use_bias:
            output += self.bias

        output = output.view(*original_shape[:-1], self.out_features)

        return output

    def regularization_loss(self, regularize_coeffs=1.0):

        coeffs_l2 = self.fourier_coeffs.pow(2).mean()
        return regularize_coeffs * coeffs_l2


class FourierKAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        num_frequencies=10,
        scale_base=1.0,
        scale_fourier=1.0,
        base_activation=torch.nn.SiLU(),
        use_bias=True,
        smooth_initialization=False,
    ):
       
        super(FourierKAN, self).__init__()

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                FourierKANLinear(
                    in_features,
                    out_features,
                    num_frequencies=num_frequencies,
                    scale_base=scale_base,
                    scale_fourier=scale_fourier,
                    base_activation=base_activation,
                    use_bias=use_bias,
                    smooth_initialization=smooth_initialization,
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

