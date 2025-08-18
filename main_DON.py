from modeltrain import modeltrain
from modelbuild import modelbuild
import argparse
import deepspeed
import os.path as osp
import torch


def add_argument():
    parser = argparse.ArgumentParser(description='CFD-CNN')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


def DON(modelnameAE = 'base-ae-kan', modelnameDON = 'base-don-kan'):
    ## model path
    dir_path = osp.dirname(osp.abspath(__file__))

    ds_args = add_argument()

    model_pathDON = osp.join(dir_path, 'Model', f'{modelnameDON}')
    total_dataDON = modelbuild(model_pathDON, ds_args)
    model_dataDON = total_dataDON.get_data()
    modelDON = modeltrain(model_dataDON, model_pathDON)

    model_pathAE = osp.join(dir_path, 'Model', f'{modelnameAE}')
    total_dataAE = modelbuild(model_pathAE, ds_args, init=False)
    model_dataAE = total_dataAE.get_data()
    modelAE = modeltrain(model_dataAE, model_pathAE)
    best_model_path = osp.join(modelAE.checkpoints_path, 'checkpoint.pth')
    modelAE.load_from_state_dict(torch.load(best_model_path,weights_only=True))

    modelDON.method1 = modelAE.method
    modelDON.method.model1 = modelAE.method.model

    modelDON.train()

    modelDON.test_inference_n()


if __name__ == '__main__':
    modelnameDON = 'base-don-kan'
    modelnameAE = 'base-ae-kan'
    DON(modelnameAE, modelnameDON)
