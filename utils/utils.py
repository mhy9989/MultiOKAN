
import torch.distributed as dist
from collections import OrderedDict
import json
from typing import Optional
from easydict import EasyDict as edict
import torch.backends.cudnn as cudnn
import random
import numpy as np
import torch
from typing import Tuple
import os
import logging
import numpy as np

def print_rank_0(message):
    """Only output in root process or single process
    """
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)

def print_log(message,if_log=True):
    if if_log:
        print_rank_0(message)
        if dist.is_initialized():
            if dist.get_rank() == 0:
                logging.info(message)
        else:
            logging.info(message)

def weights_to_cpu(state_dict: OrderedDict) -> OrderedDict:
    """Copy a model state_dict to cpu.

    Args:
        state_dict (OrderedDict): Model weights on GPU.

    Returns:
        OrderedDict: Model weights on GPU.
    """
    state_dict_cpu = OrderedDict()
    for key, val in state_dict.items():
        state_dict_cpu[key] = val.cpu()
    # Keep metadata in state_dict
    state_dict_cpu._metadata = getattr(  # type: ignore
        state_dict, '_metadata', OrderedDict())
    return state_dict_cpu

def custom_serialize(obj, indent, current_level):
    if isinstance(obj, dict):
        items = []
        for key, value in obj.items():
            key_str = f'"{key}": '
            value_str = custom_serialize(value, indent, current_level + 1)
            indent_str = ' ' * ((current_level + 1) * indent)
            items.append(f'{indent_str}{key_str}{value_str}')
        if not items:
            return '{}'
        inner = ',\n'.join(items)
        closing_indent = ' ' * (current_level * indent)
        return '{\n' + inner + '\n' + closing_indent + '}'
    elif isinstance(obj, list):
        if not obj:
            return '[]'
        items = [custom_serialize(item, indent, 0) for item in obj]
        return '[' + ', '.join(items) + ']'
    else:
        return json.dumps(obj)


def save_json(data,data_path):
    """Save json data
    """
    if dist.is_initialized():
        if dist.get_rank() == 0:
            with open(data_path, 'w') as json_file:
                json.dump(data,json_file,indent=4)
    else:
        with open(data_path, 'w') as json_file:
            json.dump(data,json_file,indent=4)



def json2Parser(json_path):
    """Load json and return a parser-like object
    Parameters
    ----------
    json_path : str
        The json file path.
    
    Returns
    -------
    args : easydict.EasyDict
        A parser-like object.
    """
    with open(json_path, 'r') as f:
        args = json.load(f)
    return edict(args)


def reduce_tensor(tensor):
    rt = tensor.data.clone()
    dist.all_reduce(rt.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return rt


def get_dist_info() -> Tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.
    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.
    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    rank, world_size = get_dist_info()
    seed = np.random.randint(2**31)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


def set_seed(seed, deterministic=True):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    cudnn.enabled = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    if deterministic:
        # torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return False
    return True


def measure_throughput(model, input_dummy):
    def get_input_dummy(input_dummy):
        def get_batch_size(max_side):
            if max_side >= 128 * 128:
                return 10, 1000
            return 100, 100
        
        input_shape = input_dummy.shape
        F = input_shape[2]
        if len(input_shape) == 5:
            H, W = input_shape[3], input_shape[4]
            L = H * W
            bs, repetitions = get_batch_size(L)
            _input = torch.rand(bs, 1, F, H, W).to(input_dummy.device)
        elif len(input_shape) == 4:
            L = input_shape[3]
            bs, repetitions = get_batch_size(L)
            _input = torch.rand(bs, 1, F, L).to(input_dummy.device)
        return _input, bs, repetitions

    if isinstance(input_dummy, tuple):
        input_dummy = list(input_dummy)
        input_dummy[0], bs, repetitions = get_input_dummy(input_dummy[0])
        input_dummy = tuple(input_dummy)
    else:
        input_dummy, bs, repetitions = get_input_dummy(input_dummy)
    total_time = 0
    with torch.no_grad():
        for _ in range(repetitions):
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            if isinstance(input_dummy, tuple):
                _ = model(*input_dummy)
            else:
                _ = model(input_dummy)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) / 1000
            total_time += curr_time
    Throughput = (repetitions * bs) / total_time
    return Throughput


def output_namespace(namespace):
    configs = namespace.__dict__
    message = ''
    ban_list = ["setup_config", "data_config", "noise_config", "optim_config", "sched_config", "model_config", "ds_config"]
    for k, v in configs.items():
        if k not in ban_list:
            message += '\n' + k + ': \t' + str(v) + '\t'
    return message


def add_gaussian_noise(
    inputs: np.ndarray,
    noise_scale: float = 0.1,
    sample_ratio: float = 1.0,
    mask_ratio: float = 0.1,
    samlpes_seed: Optional[int] = None,
    mask_seed: Optional[int] = None,
    noise_seed: Optional[int] = None,
):
    """
    Add Gaussian noise based on the locate value to the specified dimension (dim=2)
    
    Args:
        inputs: Input ndarray (B, 1, F, L)
        noise_scale: Noise intensity coefficient (default 10%)
        sample_ratio: Proportion of samples to select (default 100%)
        mask_ratio: Proportion of points to select (default 10%)
        samlpes_seed: Random seed for samples selection
        mask_seed: Random seed for point selection
        noise_seed: Random seed for noise generation
    
    Returns:
        New tensor with added noise
    """
    # Parameter validation
    assert 0.0 <= noise_scale <= 1.0, "noise_scale must be between 0 and 1"
    assert 0.0 <= mask_ratio <= 1.0, "mask_ratio must be between 0 and 1"
    assert 0.0 <= sample_ratio <= 1.0, "sample_ratio must be between 0 and 1"

    B, F, L = inputs.shape
    
    # Calculate the number of samples to select
    J = int(round(B * sample_ratio))
    
    # Calculate the number of points to select
    K = int(round(L * mask_ratio))

    if K == 0 or J == 0 or noise_scale == 0:
        print_log("No noise added")
        return inputs.copy()
    # Generate indices for point selection (precise selection)
    sample_generator = np.random.default_rng(samlpes_seed)
    selected_sample_indices = sample_generator.permutation(B)[:J]
    
    # Generate indices for point selection (precise selection)
    mask_generator = np.random.default_rng(mask_seed)
    selected_points_indices = mask_generator.permutation(L)[:K]
    
    # Generate noise data
    noise_generator = np.random.default_rng(noise_seed)
    noise = noise_generator.normal(scale=noise_scale, size=(J, F, K))
    noise = np.clip(noise,-1,1)
    
    # Apply noise
    output_tensor = inputs.copy().reshape(B, F, L)
    ixgrid = np.ix_(selected_sample_indices, range(F), selected_points_indices)
    output_tensor[ixgrid] *= (1.0 + noise)
    
    return output_tensor.reshape(B, F, L)