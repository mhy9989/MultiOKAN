# Based on this: https://github.com/lgy112112/ikan/blob/main/ikan/WaveletKAN.py
# Based on this: https://github.com/zavareh1/Wav-KAN/blob/main/KAN.py
import torch
import torch.nn.functional as F
import math

class WaveletKANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        wavelet_type='mexican_hat',
        scale_base=1.0,
        scale_wavelet=1.0,
        base_activation=torch.nn.SiLU(),
        use_bias=True,
    ):
        super(WaveletKANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.wavelet_type = wavelet_type
        self.scale_base = scale_base 
        self.scale_wavelet = scale_wavelet 
        self.base_activation = base_activation
        self.use_bias = use_bias  

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))

        self.wavelet_weights = torch.nn.Parameter(
            torch.Tensor(out_features, in_features)
        )

        self.scale = torch.nn.Parameter(torch.ones(out_features, in_features))
        self.translation = torch.nn.Parameter(torch.zeros(out_features, in_features))

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
            std = self.scale_wavelet / math.sqrt(self.in_features)
            self.wavelet_weights.uniform_(-std, std)

        torch.nn.init.ones_(self.scale)
        torch.nn.init.zeros_(self.translation)

        if self.use_bias:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.base_weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def wavelet_transform(self, x):
        batch_size = x.size(0)

        x_expanded = x.unsqueeze(1)

        scale = self.scale  # (out_features, in_features)
        translation = self.translation  # (out_features, in_features)

        scale_expanded = scale.unsqueeze(0)
        translation_expanded = translation.unsqueeze(0)

        x_scaled = (x_expanded - translation_expanded) / scale_expanded  # (batch_size, out_features, in_features)

        if self.wavelet_type == 'mexican_hat':
            term1 = (x_scaled ** 2 - 1)
            term2 = torch.exp(-0.5 * x_scaled ** 2)
            wavelet = term1 * term2
        elif self.wavelet_type == 'morlet':
            omega0 = 5.0 
            real = torch.cos(omega0 * x_scaled)
            envelope = torch.exp(-0.5 * x_scaled ** 2)
            wavelet = envelope * real
        elif self.wavelet_type == 'dog':
            wavelet = -x_scaled * torch.exp(-0.5 * x_scaled ** 2)
        elif self.wavelet_type == 'meyer':
            pi = math.pi
            v = torch.abs(x_scaled)
            wavelet = torch.sin(pi * v) * self.meyer_aux(v)
        elif self.wavelet_type == 'shannon':
            pi = math.pi
            sinc = torch.sinc(x_scaled / pi)
            window = torch.hamming_window(
                x_scaled.size(-1), periodic=False, dtype=x_scaled.dtype, device=x_scaled.device
            )
            wavelet = sinc * window
        else:
            raise ValueError(f"Unsupported wavelet type: {self.wavelet_type}")

        wavelet_weights_expanded = self.wavelet_weights.unsqueeze(0)

        wavelet_output = (wavelet * wavelet_weights_expanded).sum(dim=2)  # (batch_size, out_features)

        return wavelet_output

    def meyer_aux(self, v):
        pi = math.pi

        def nu(t):
            return t ** 4 * (35 - 84 * t + 70 * t ** 2 - 20 * t ** 3)

        cond1 = v <= 0.5
        cond2 = (v > 0.5) & (v < 1.0)

        result = torch.zeros_like(v)
        result[cond1] = 1.0
        result[cond2] = torch.cos(pi / 2 * nu(2 * v[cond2] - 1))

        return result

    def forward(self, x: torch.Tensor):
        original_shape = x.shape

        x = x.view(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)

        wavelet_output = self.wavelet_transform(x)  # (batch_size, out_features)

        output = base_output + wavelet_output

        if self.use_bias:
            output += self.bias

        output = output.view(*original_shape[:-1], self.out_features)

        return output

    def regularization_loss(self, regularize_coeffs=1.0):
        coeffs_l2 = self.wavelet_weights.pow(2).mean()
        return regularize_coeffs * coeffs_l2


class WaveletKAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        wavelet_type='mexican_hat',
        scale_base=1.0,
        scale_wavelet=1.0,
        base_activation=torch.nn.SiLU(),
        use_bias=True,
    ):
        super(WaveletKAN, self).__init__()

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                WaveletKANLinear(
                    in_features,
                    out_features,
                    wavelet_type=wavelet_type,
                    scale_base=scale_base,
                    scale_wavelet=scale_wavelet,
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
