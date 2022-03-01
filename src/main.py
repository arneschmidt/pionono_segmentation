# TODO
# calculate metrics for complete test set instead of single batches


import os
import argparse
import utils.globals

from data import get_data_supervised
from model_handler import ModelHandler
from utils.logging import start_logging


def main():
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    start_logging()

    model_handler = ModelHandler()
    mode = utils.globals.config['model']['mode']

    print("Mode: " + mode)
    if mode == 'train':
        trainloader = get_data_supervised(split='train')
        validateloader = get_data_supervised(split='val')
        testloader = get_data_supervised(split='test')

        model_handler.train(trainloader, validateloader)
        model_handler.test(testloader)
    elif mode == 'test':
        testloader = get_data_supervised(split='test')
        model_handler.test(testloader)
    elif mode == 'predict':
        testloader = get_data_supervised(split='test')
        model_handler.predict(testloader)
    else:
        print('Choose a valid model mode: "train", "test", "predict".')


if __name__ == "__main__":
    print('Load configuration')

    parser = argparse.ArgumentParser(description="Cancer Classification")
    parser.add_argument("--default_config", "-dc", type=str, default="./config.yaml",
                        help="Config path (yaml file expected) to default config.")
    parser.add_argument("--experiment_folder", "-ef", type=str, default="None",
                        help="Config path to experiment folder. Parameters will override defaults. Optional.")
    args = parser.parse_args()
    utils.globals.init_global_config(args)
    main()