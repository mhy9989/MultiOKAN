import argparse
import deepspeed
from main_AE import AE
from main_DON import DON
from inference_AE import inference_AE
from predata import presetting, predata


def add_argument():
    parser = argparse.ArgumentParser(description='CFD-CNN')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


def AE_DON(modelnameAE = 'base-ae-kan',modelnameDON = 'base-don-kan'):
    presetting(modelnameAE)
    AE(modelnameAE)
    inference_AE(modelnameAE)
    predata(modelnameAE,modelnameDON)
    DON(modelnameAE,modelnameDON)


if __name__ == '__main__':
    modelnameAE = 'base-ae-kan'
    modelnameDON = 'base-don-kan'
    AE_DON(modelnameAE, modelnameDON)
