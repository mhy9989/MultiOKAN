from .template import Template
from .l_deeponet import L_DeepONet
from .l_deepokan import L_DeepOKan

method_maps = {
    'l-deeponet': L_DeepONet,
    'l-deepokan': L_DeepOKan
}

__all__ = [
    'method_maps', 'Template', 
    'l-deeponet', 'l-deepokan'
]