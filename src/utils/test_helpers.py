import numpy as np
import utils.globals as globals

from sklearn.metrics import accuracy_score, jaccard_score

def dice_coef_binary(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    smooth = 0.0001
    return (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)

def dice_coef_multilabel(y_true, y_pred):
    class_no = globals.config['data']['class_no']
    dice_per_class = []
    for index in range(class_no):
        dice_per_class.append(dice_coef_binary(y_true == index, y_pred == index))

    return np.array(dice_per_class)

def segmentation_scores(label_trues, label_preds, metric_names):
    '''
    :param label_trues:
    :param label_preds:
    :param n_class:
    :return:
    '''
    results = {}
    class_no = globals.config['data']['class_no']
    class_names = globals.config['data']['class_names']
    ignore_last_class = globals.config['data']['ignore_last_class']

    assert len(label_trues) == len(label_preds)

    # label_preds = np.asarray(label_preds, dtype='int8').copy().flatten()
    # label_trues = np.asarray(label_trues, dtype='int8').copy().flatten()
    label_preds = np.array(label_preds, dtype='int8')
    label_trues = np.array(label_trues, dtype='int8')

    if ignore_last_class:
        label_preds = label_preds[label_trues!=class_no]
        label_trues = label_trues[label_trues!=class_no]

    dice_per_class = dice_coef_multilabel(label_trues, label_preds)

    results['macro_dice'] = dice_per_class.mean()

    intersection = (label_preds == label_trues).sum(axis=None)
    sum_ = 2 * np.prod(label_preds.shape)
    results['micro_dice'] = ((2 * intersection + 1e-6) / (sum_ + 1e-6))

    for class_id in range(class_no):
        results['dice_class_' + str(class_id) + '_' + class_names[class_id]] = dice_per_class[class_id]

    results['accuracy'] = accuracy_score(label_trues, label_preds)
    results['miou'] = jaccard_score(label_trues, label_preds, average="macro") # same as IoU!

    for metric in metric_names:
        assert metric in results.keys()

    return results