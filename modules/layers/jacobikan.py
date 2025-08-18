# Based on this: https://github.com/SpaceLearner/JacobiKAN/blob/main/JacobiKANLayer.py
# Based on this: https://github.com/lgy112112/ikan/blob/main/ikan/JacobiKAN.py

import torch
import torch.nn.functional as F
import math

class JacobiKANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        degree=5,
        a=1.0,
        b=1.0,
        scale_base=1.0,
        scale_jacobi=1.0,
        base_activation=torch.nn.SiLU(),
        use_bias=True,
    ):
        super(JacobiKANLinear, self).__init__()
        self.in_features = in_features 
        self.out_features = out_features 
        self.degree = degree
        self.a = a 
        self.b = b 
        self.scale_base = scale_base 
        self.scale_jacobi = scale_jacobi
        self.base_activation = base_activation 
        self.use_bias = use_bias 

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))

        self.jacobi_coeffs = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, degree + 1)
        )

        if self.use_bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)

        with torch.no_grad():
            std = self.scale_jacobi / (self.in_features * math.sqrt(self.degree + 1))
            self.jacobi_coeffs.normal_(mean=0.0, std=std)

        if self.use_bias:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.base_weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def jacobi_polynomials(self, x: torch.Tensor):
        x = torch.tanh(x)

        batch_size, in_features = x.size()
        jacobi = torch.zeros(batch_size, in_features, self.degree + 1, device=x.device)
        jacobi[:, :, 0] = 1.0  # P_0(x) = 1

        if self.degree >= 1:
            jacobi[:, :, 1] = 0.5 * ((2 * (self.a + 1)) * x + (self.a - self.b))  # P_1(x)

        for n in range(2, self.degree + 1):
            n_minus_1 = jacobi[:, :, n - 1]  # P_{n-1}(x)
            n_minus_2 = jacobi[:, :, n - 2]  # P_{n-2}(x)

            k = n - 1
            alpha_n = 2 * k * (k + self.a + self.b) * (2 * k + self.a + self.b - 2)
            beta_n = (2 * k + self.a + self.b - 1) * (self.a ** 2 - self.b ** 2)
            gamma_n = (2 * k + self.a + self.b - 2) * (2 * k + self.a + self.b - 1) * (2 * k + self.a + self.b)
            delta_n = 2 * (k + self.a - 1) * (k + self.b - 1) * (2 * k + self.a + self.b)

            A = (beta_n + alpha_n * x) / gamma_n
            B = delta_n / gamma_n

            next_jacobi = A * n_minus_1 - B * n_minus_2
            jacobi = torch.cat([jacobi[:, :, :n], next_jacobi.unsqueeze(2)], dim=2)

        return jacobi


    def forward(self, x: torch.Tensor):
        original_shape = x.shape

        x = x.view(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)

        P_n = self.jacobi_polynomials(x) 

        jacobi_output = torch.einsum('bik,oik->bo', P_n, self.jacobi_coeffs)

        output = base_output + jacobi_output

        if self.use_bias:
            output += self.bias

        output = output.view(*original_shape[:-1], self.out_features)

        return output

    def regularization_loss(self, regularize_coeffs=1.0):
        coeffs_l2 = self.jacobi_coeffs.pow(2).mean()
        return regularize_coeffs * coeffs_l2

class JacobiKAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        degree=5,
        a=1.0,
        b=1.0,
        scale_base=1.0,
        scale_jacobi=1.0,
        base_activation=torch.nn.SiLU(),
        use_bias=True,
    ):
        super(JacobiKAN, self).__init__()

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                JacobiKANLinear(
                    in_features,
                    out_features,
                    degree=degree,
                    a=a,
                    b=b,
                    scale_base=scale_base,
                    scale_jacobi=scale_jacobi,
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
