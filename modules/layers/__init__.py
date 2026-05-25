from .kan import KANLinear, KAN
from .chebykan import ChebyKANLinear, ChebyKAN
from .fourierkan import FourierKANLinear, FourierKAN
from .taylorkan import TaylorKANLinear, TaylorKAN
from .waveletkan import WaveletKANLinear, WaveletKAN
from .fastkan import FastKANLayer, FastKAN
from .rbfkan import RBFKANLayer, RBFKAN
from .conv import BasicConv2d, ConvSC, sampling_generator
from .convkan import ConvKANSC
from .kan_convs import sampling_generator_kan
from .gramkan import GRAMLayer, GRAMKAN
from .gate import GatedFusion, CrossStitchUnit, DynamicFusion

__all__ = [
    'KANLinear', 'KAN', 'ChebyKAN', 'ChebyKANLinear', 
    'FourierKAN', 'FourierKANLinear',
    'TaylorKAN', 'TaylorKANLinear',
    'WaveletKAN','WaveletKANLinear',
    'FastKAN','FastKANLayer',
    'RBFKAN','RBFKANLayer',
    'get_kan','get_kan_layer',
    'BasicConv2d', 'ConvSC', 'sampling_generator',
    'ConvKANSC', 'sampling_generator_kan',
    'GRAMKAN', 'GRAMLayer', 'GatedFusion', 'CrossStitchUnit', 'DynamicFusion'
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

def get_kan_layer(kan_type):
    kan_type = kan_type.lower()
    if kan_type == "base":
        kan_layer = KANLinear
    elif kan_type == "cheby":
        kan_layer = ChebyKANLinear
    elif kan_type == "fourier":
        kan_layer = FourierKANLinear
    elif kan_type == "taylor":
        kan_layer = TaylorKANLinear
    elif kan_type == "wave":
        kan_layer = WaveletKANLinear
    elif kan_type == "fast":
        kan_layer = FastKANLayer
    elif kan_type == "rbf":
        kan_layer = RBFKANLayer
    elif kan_type == "gram":
        kan_layer = GRAMLayer
    else:
        raise ValueError("Unknown kan_type")

    return kan_layer

