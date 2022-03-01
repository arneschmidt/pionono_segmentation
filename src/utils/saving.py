import os
import imageio
import errno
import csv
import torch
import mlflow
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image

import utils.globals as globals

def save_model(model):
    model_dir = 'models'
    dir = os.path.join(globals.config['logging']['experiment_folder'], model_dir)
    os.makedirs(dir, exist_ok=True)
    out_path = os.path.join(dir, 'best_model.pth')
    torch.save(model, out_path)
    print('Best Model saved!')

def save_test_images(test_imgs:torch.Tensor, test_preds: np.array, test_labels: np.array, test_name: np.array, mode: str):
    if mode != 'predict':
        folder_name = mode
    else:
        folder_name = 'test'

    visual_dir = 'qualitative_results/' + mode
    dir = os.path.join(globals.config['logging']['experiment_folder'], visual_dir)
    os.makedirs(dir, exist_ok=True)

    h, w = np.shape(test_preds)

    test_preds = np.asarray(test_preds, dtype=np.uint8)
    test_labels = np.asarray(test_labels, dtype=np.uint8)

    out_path_img = os.path.join(dir, 'img_' + test_name)
    save_image(test_imgs, out_path_img)

    test_pred_rgb = convert_classes_to_rgb(test_preds, h, w)
    out_path_pred = os.path.join(dir, 'pred_' + test_name)
    imageio.imsave(out_path_pred, test_pred_rgb)

    if mode != 'predict':
        test_label_rgb = convert_classes_to_rgb(test_labels, h, w)
        out_path_gt = os.path.join(dir, 'gt_' + test_name)
        imageio.imsave(out_path_gt, test_label_rgb)
        mlflow.log_artifacts(dir, visual_dir)
    else:
        vis_images = globals.config['data']['visualize_images']['test']
        if test_name in vis_images or vis_images=='all':
            mlflow.log_artifact(out_path_img, visual_dir)
            mlflow.log_artifact(out_path_pred, visual_dir)


def save_image_color_legend():
    visual_dir = 'qualitative_results/'
    dir = os.path.join(globals.config['logging']['experiment_folder'], visual_dir)
    os.makedirs(dir, exist_ok=True)
    class_no = globals.config['data']['class_no']
    class_names = globals.config['data']['class_names']

    fig = plt.figure()

    size = 100

    for class_id in range(class_no):
        # out_img[size*class_id:size*(class_id+1),:,:] = convert_classes_to_rgb(np.ones(size,size,3)*class_id, size,size)
        out_img = convert_classes_to_rgb(np.ones(shape=[size,size])*class_id, size,size)
        ax = fig.add_subplot(1, class_no, class_id+1)
        ax.imshow(out_img)
        ax.set_title(class_names[class_id])
        ax.axis('off')
    plt.savefig(dir + 'legend.png')
    mlflow.log_artifact(dir + 'legend.png', 'qualitative_results')


def convert_classes_to_rgb(seg_classes, h, w):

    seg_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    class_no = globals.config['data']['class_no']

    colors = [[0,179,255], [153,0,0], [255,102,204], [0,153,51], [153,0,204]]

    for class_id in range(class_no):
        seg_rgb[:, :, 0][seg_classes == class_id] = colors[class_id][0]
        seg_rgb[:, :, 1][seg_classes == class_id] = colors[class_id][1]
        seg_rgb[:, :, 2][seg_classes == class_id] = colors[class_id][2]

    return seg_rgb


def save_results(results):
    results_dir = 'quantitative_results'
    dir = os.path.join(globals.config['logging']['experiment_folder'], results_dir)
    os.makedirs(dir, exist_ok=True)
    out_path = os.path.join(dir, 'results.csv')

    with open(out_path, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in results.items():
            writer.writerow([key, value])