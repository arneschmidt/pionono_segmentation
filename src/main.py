# TODO
# calculate metrics for complete test set instead of single batches


import os
import argparse

from data import get_data_supervised
from utils.globals import init_global_config
from model_handler import ModelHandler
from utils.logging import start_logging


def main():
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

    start_logging()

    # load data
    trainloader, validateloader, testloader = get_data_supervised()

    # load and train the model
    model_handler = ModelHandler()
    model_handler.train(trainloader, validateloader)
    model_handler.test(testloader)


if __name__ == "__main__":
    print('Load configuration')

    parser = argparse.ArgumentParser(description="Cancer Classification")
    parser.add_argument("--default_config", "-dc", type=str, default="./config.yaml",
                        help="Config path (yaml file expected) to default config.")
    parser.add_argument("--experiment_folder", "-ef", type=str, default="None",
                        help="Config path to experiment folder. Parameters will override defaults. Optional.")
    args = parser.parse_args()
    init_global_config(args)
    main()
