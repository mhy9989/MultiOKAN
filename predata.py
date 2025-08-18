import numpy as np
import os
import json
from easydict import EasyDict as edict

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return False
    return True
    
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

def save_json(data, filepath, indent=2):
    json_str = custom_serialize(data, indent, 0)
    with open(filepath, 'w') as f:
        f.write(json_str)


def json2Parser(json_path):
    with open(json_path, 'r') as f:
        args = json.load(f)
    return edict(args)

def presetting(in_path = "base-ae-kan"): # wirte AE config to settings

    setting_path = f"./Model/{in_path}/checkpoints/settings.json"

    args = json2Parser(setting_path)
    features = len(args.data_config.data_select)
    data = np.load(args.data_config.org_path)

    data_all = np.round(np.linspace(0, args.data_config.total_after, 
                                      args.data_config.data_after_num+1,
                                      endpoint=True)).astype(int)
    data_matrix = data[:args.data_config.data_num, data_all].astype(np.float64).copy()
    data_matrix = data_matrix[:, :, args.data_config.data_select] 

    print(data_matrix.shape)
    mean = [[]]
    std = [[]]
    min = [[]]
    max = [[]]

    for i in range(features):
        mean[0].append(data_matrix[:,:,i].mean())
        std[0].append(data_matrix[:,:,i].std())
        max[0].append(data_matrix[:,:,i].max())
        min[0].append(data_matrix[:,:,i].min())

    args.data_config.data_mean=mean
    args.data_config.data_std=std
    args.data_config.data_max=max
    args.data_config.data_min=min

    save_json(args, setting_path)


def predata(in_path = "base-ae-kan",out_path = "base-don-kan"): # wirte DON config to settings

    input_setting_path = f"./Model/{in_path}/checkpoints/settings.json"
    output_setting_path = f"./Model/{out_path}/checkpoints/settings.json"
    data = np.load(f"./Model/{in_path}/saved/AE/AE_preds.npy")

    print(data.shape)

    op = f"./Model/{out_path}/data"
    check_dir(op)
    np.save(f"{op}/AE_lam.npy", data)

    in_args = json2Parser(input_setting_path)
    out_args = json2Parser(output_setting_path)
    features = len(in_args.data_config.data_select)
    mean = [[],[]]
    std = [[],[]]
    min = [[],[]]
    max = [[],[]]

    mean[0]=in_args.data_config.data_mean[0]
    std[0]=in_args.data_config.data_std[0]
    max[0]=in_args.data_config.data_max[0]
    min[0]=in_args.data_config.data_min[0]

    for i in range(features):
        mean[1].append(data[:,:,i].mean())
        std[1].append(data[:,:,i].std())
        max[1].append(data[:,:,i].max())
        min[1].append(data[:,:,i].min())
        
    out_args.setup_config.seed=in_args.setup_config.seed
    out_args.data_config.data_mean=mean
    out_args.data_config.data_std=std
    out_args.data_config.data_max=max
    out_args.data_config.data_min=min
    out_args.data_config.lam_path=f"{op}/AE_lam.npy"
    out_args.data_config.org_path=in_args.data_config.org_path
    out_args.data_config.mesh_path=in_args.data_config.mesh_path
    out_args.data_config.train_ratio=in_args.data_config.train_ratio
    out_args.data_config.train_seed=in_args.data_config.train_seed
    out_args.data_config.total_after=in_args.data_config.total_after
    out_args.data_config.data_after_num=in_args.data_config.data_after_num
    out_args.data_config.dt=in_args.data_config.dt

    save_json(out_args, output_setting_path)

if __name__ == '__main__':
    in_path = "base-ae-kan"
    out_path = "base-don-kan"
    #presetting(in_path = "base-ae-kan") # wirte AE config to settings
    predata(in_path, out_path)           # wirte DON config to settings
