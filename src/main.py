# TODO
# make correct cross validation
# implement staple
# IDEAS: use F * (1 + z) instead of [F, z]
#
# today: set up experiments with best performing model, find expressive test images
# tomorrow: implement model freezing


import os
import argparse

import torch

from data import get_data_supervised
from utils.globals import init_global_config
import utils.globals
from model_handler import ModelHandler
from utils.mlflow_logger import start_logging


def main():
    print(os.curdir)
    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(3)
    #os.environ['CUDA_LAUNCH_BLOCKING'] = str(1)

    start_logging()

    # load data
    trainloader, validateloader, testloader, annotators = get_data_supervised()

    # load and train the model
    model_handler = ModelHandler(annotators)
    model_handler.train(trainloader, validateloader)
    model_handler.test(testloader)
    # if globals.config['data']['sr_experiment']:
    #     model_handler.evaluate_sr(testloader)


if __name__ == "__main__":
    print('Load configuration')

    parser = argparse.ArgumentParser(description="Cancer Classification")
    parser.add_argument("--config", "-c", type=str, default="./config.yaml",
                        help="Config path (yaml file expected) to default config.")
    parser.add_argument("--dataset_config", "-dc", type=str, default="./dataset_dependent/gleason19/dataset_config_crowd_crossval0.yaml",
                        help="Config path (yaml file expected) to default config.")
    parser.add_argument("--experiment_folder", "-ef", type=str, default="None",
                        help="Config path to experiment folder. Parameters will override defaults. Optional.")
    args = parser.parse_args()
    init_global_config(args)
    config = utils.globals.config
    torch.manual_seed(config['model']['seed'])
    main()