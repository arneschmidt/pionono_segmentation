import torch
import numpy as np


def segmentation_scores(label_trues, label_preds, n_class):
    '''
    :param label_trues:
    :param label_preds:
    :param n_class:
    :return:
    '''
    assert len(label_trues) == len(label_preds)

    label_preds = label_preds[label_trues!=n_class]
    label_trues = label_trues[label_trues!=n_class]

    label_preds = np.asarray(label_preds, dtype='int8').copy()
    label_trues = np.asarray(label_trues, dtype='int8').copy()

    ignore_class = np.ones_like(label_preds) * n_class
    label_preds = np.where(label_trues==n_class, ignore_class, label_preds)
    intersection = np.where(label_preds == label_trues, label_preds, ignore_class)

    (area_intersection, _) = np.histogram(intersection, bins=n_class, range=(0, n_class))
    (area_pred, _) = np.histogram(label_preds, bins=n_class, range=(0, n_class))
    (area_lab, _) = np.histogram(label_trues, bins=n_class, range=(0, n_class))

    area_sum = area_pred + area_lab

    dice = ((2 * area_intersection + 1e-6) / (area_sum + 1e-6))
    macro_dice = dice.mean()

    intersection = (label_preds == label_trues).sum(axis=None)
    sum_ = 2 * np.prod(label_preds.shape)
    micro_dice = ((2 * intersection + 1e-6) / (sum_ + 1e-6))

    return dice, macro_dice, micro_dice


