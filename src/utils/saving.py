import os
import imageio
import errno
import csv
import torch
import numpy as np

from src.utils.globals import config

def save_model(model):
    model_dir = 'models'
    dir = os.path.join(config['logging']['experiment_folder'], model_dir)
    os.makedirs(dir, exist_ok=True)
    out_path = os.path.join(dir, '/best_model.pth')
    torch.save(model, out_path)
    print('Best Model saved!')

def save_test_image(test_preds: np.array, testlabel: np.array, testname: np.array):
    visual_dir = 'qualitative_results'
    dir = os.path.join(config['logging']['experiment_folder'], visual_dir)
    os.makedirs(dir, exist_ok=True)

    b, h, w = np.shape(testlabel)

    testoutput_original = np.asarray(test_preds, dtype=np.uint8) + 1
    segmentation_map = 2 * np.zeros((h, w, 3), dtype=np.uint8)

    for i, test_out in enumerate(testoutput_original):
        # multi class for breast cancer
        segmentation_map[:, :, 0][test_out == 1] = 153
        segmentation_map[:, :, 1][test_out == 1] = 0
        segmentation_map[:, :, 2][test_out == 1] = 0

        segmentation_map[:, :, 0][test_out == 2] = 255
        segmentation_map[:, :, 1][test_out == 2] = 102
        segmentation_map[:, :, 2][test_out == 2] = 204

        segmentation_map[:, :, 0][test_out == 3] = 0
        segmentation_map[:, :, 1][test_out == 3] = 153
        segmentation_map[:, :, 2][test_out == 3] = 51

        segmentation_map[:, :, 0][test_out == 4] = 153
        segmentation_map[:, :, 1][test_out == 4] = 0
        segmentation_map[:, :, 2][test_out == 4] = 204

        segmentation_map[:, :, 0][test_out == 5] = 0
        segmentation_map[:, :, 1][test_out == 5] = 179
        segmentation_map[:, :, 2][test_out == 5] = 255

        segmentation_map[:, :, 0][testlabel[i] == 0] = 0
        segmentation_map[:, :, 1][testlabel[i] == 0] = 0
        segmentation_map[:, :, 2][testlabel[i] == 0] = 0


        out_path = os.path.join(dir, 'pred_seg_' + testname[i])
        imageio.imsave(out_path, segmentation_map)

def save_results(results):
    results_dir = 'quantitative_results'
    dir = os.path.join(config['logging']['experiment_folder'], results_dir)
    os.makedirs(dir, exist_ok=True)
    out_path = os.path.join(dir, 'results.csv')

    with open(out_path, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in results.items():
            writer.writerow([key, value])

def save_model_path(pretrained_net, labels_mode, epochs, train_batchsize):

    save_model_name = pretrained_net + '_pretrained_Unet_' + labels_mode + '_e' + str(epochs) + '_b_size' + str(train_batchsize)
    saved_information_path = '/home/mlopez/segmentation_Lee/Results_segmentation_augmentation_weighted/'
    #
    try:
        os.mkdir(saved_information_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    #
    #
    saved_information_path = saved_information_path + 'Results_' + save_model_name
    #
    try:
        os.mkdir(saved_information_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    #
    saved_model_path = saved_information_path + '/trained_models'
    try:
        os.mkdir(saved_model_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    #
    #
    print('The current model is:')
    #
    print(save_model_name)
    #
    print('\n')
    #
    #writer = SummaryWriter('../../Results/Log_UNet_gold/' + save_model_name)
    return saved_information_path, save_model_name
