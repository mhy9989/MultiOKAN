from .kan import KANLinear, KAN
from .chebykan import ChebyKAN
from .fourierkan import FourierKAN
from .taylorkan import TaylorKAN
from .waveletkan import WaveletKAN
from .fastkan import FastKAN
from .rbfkan import RBFKAN
from .conv import BasicConv2d, ConvSC, sampling_generator
from .convkan import ConvKANSC
from .kan_convs import sampling_generator_kan
from .gramkan import GRAMKAN


__all__ = [
    'KANLinear', 'KAN', 'ChebyKAN', 'FourierKAN', 
    'TaylorKAN', 'WaveletKAN',
    'FastKAN','RBFKAN','get_kan',
    'BasicConv2d', 'ConvSC', 'sampling_generator',
    'ConvKANSC', 'sampling_generator_kan',
    'GRAMKAN',
]

def get_kan(kan_type, de_layer, base_activation, kan_config):
    kan_type = kan_type.lower()
    if kan_type == "base":
        kan_net = KAN
    elif kan_type == "cheby":
        kan_net = ChebyKAN
    elif kan_type == "fourier":
        kan_net = FourierKAN
    elif kan_type == "taylor":
        kan_net = TaylorKAN
    elif kan_type == "wave":
        kan_net = WaveletKAN
    elif kan_type == "fast":
        kan_net = FastKAN
    elif kan_type == "rbf":
        kan_net = RBFKAN
    elif kan_type == "gram":
        kan_net = GRAMKAN
    else:
        raise ValueError("Unknown kan_type")

    return kan_net(de_layer, base_activation=base_activation, **kan_config)
