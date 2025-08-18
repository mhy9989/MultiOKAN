from modeltrain import modeltrain
from modelbuild import modelbuild
import os
import argparse
import deepspeed

def add_argument():
    parser = argparse.ArgumentParser(description='CFD-CNN')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args

def inference_AE(modelname = 'base-ae-kan'):
    ## model name
    mode = "inference"
    #tt_num = 11*1000
    infer_num = [-1] #range(tt_num)
    #infer_num = range(1001) #range(-50,-10)
    min_max_delt=None
    if_AE="en"
    ## model path
    dir_path = os.path.dirname(os.path.abspath(__file__))

    ds_args = add_argument()
    model_path = os.path.join(dir_path, 'Model', f'{modelname}')
    total_data = modelbuild(model_path, ds_args, mode)
    model_data = total_data.get_data()
    model = modeltrain(model_data, model_path, mode, infer_num = infer_num)
    model.test_inference(min_max_delt, if_AE)



if __name__ == '__main__':
    inference_AE()
