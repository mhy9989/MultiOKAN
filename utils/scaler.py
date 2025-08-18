import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils import print_log


def get_torch_scaler(Scaler):
    if isinstance(Scaler, StandardScaler):
        torch_scaler = torch_StandardScaler(Scaler.mean_, Scaler.scale_)
    elif isinstance(Scaler, MinMaxScaler):
        torch_scaler = torch_MinMaxScaler(Scaler.min_, Scaler.scale_)
    elif isinstance(Scaler, NoneScaler):
        torch_scaler = NoneScaler()
    else:
        raise ValueError("Unknown scaler type")
    return torch_scaler


def get_scaler(args, num, type=0):
    if args.data_scaler[num] == "Standard":
        scaler = StandardScaler()
        scaler.mean_= args.data_mean[num][type]
        scaler.scale_ = args.data_std[num][type]
    elif args.data_scaler[num] == "MinMax":
        scaler= MinMaxScaler()
        custom_min = -1.0
        custom_max = 1.0
        scale = (custom_max - custom_min) / (args.data_max[num][type] - args.data_min[num][type])
        scaler.scale_ = scale
        scaler.min_= custom_min - args.data_min[num][type] * scale
    elif args.data_scaler[num] == "None":
        scaler = NoneScaler()
    else:
        print_log(f"Error data_scaler type: {args.data_scaler[num]}")
        raise EOFError
    return scaler


class NoneScaler():
    def __init__(self):
        return

    def transform(self, data):
        return data
    
    def inverse_transform(self, data):
        return data


class torch_StandardScaler:

    def __init__(self, mean=None, std=None):
        """Standard Scaler.
        The class can be used to normalize PyTorch Tensors using native functions. The module does not expect the
        tensors to be of any specific shape; as long as the features are the last dimension in the tensor, the module
        will work fine.
        :param mean: The mean of the features. The property will be set after a call to fit.
        :param std: The standard deviation of the features. The property will be set after a call to fit.
        """
        self.mean = mean
        self.std = std

    def fit(self, values):
        dims = list(range(values.dim() - 1))
        self.mean = torch.mean(values, dim=dims)
        self.std = torch.std(values, dim=dims)

    def transform(self, values):
        return (values - self.mean) / self.std
    
    def inverse_transform(self, values):
        return (values * self.std) + self.mean

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)
    

class torch_MinMaxScaler:

    def __init__(self, min=None, scale=None):
        """Minmax Scaler.
        The class can be used to normalize PyTorch Tensors using native functions. The module does not expect the
        tensors to be of any specific shape; as long as the features are the last dimension in the tensor, the module
        will work fine.
        :param mean: The mean of the features. The property will be set after a call to fit.
        :param std: The standard deviation of the features. The property will be set after a call to fit.
        """
        self.min = min
        self.scale = scale

    def fit(self, values):
        dims = list(range(values.dim() - 1))
        data_max = torch.max(values, dim=dims)
        data_min = torch.min(values, dim=dims)
        data_range = data_max - data_min
        self.scale = 1 / data_range
        self.min = - data_min * self.scale

    def transform(self, values):
        return (values * self.scale) + self.min
    
    def inverse_transform(self, values):
        return (values - self.min) / self.scale

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)