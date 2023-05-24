import numpy as np
import utils.globals as globals

from sklearn.metrics import accuracy_score, jaccard_score, f1_score, cohen_kappa_score

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

def segmentation_scores(label_trues, label_preds, shortened):
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
    ignore_last_class_only_for_testing = globals.config['data']['ignore_last_class_only_for_testing']

    assert len(label_trues) == len(label_preds)

    # label_preds = np.asarray(label_preds, dtype='int8').copy().flatten()
    # label_trues = np.asarray(label_trues, dtype='int8').copy().flatten()
    label_preds = np.array(label_preds, dtype='int8')
    label_trues = np.array(label_trues, dtype='int8')

    if ignore_last_class:
        label_trues = label_trues[label_trues!=class_no+1]

    if ignore_last_class_only_for_testing:
        label_preds = label_preds[label_trues!=class_no]
        label_trues = label_trues[label_trues!=class_no]
        nc_class = np.ones_like(label_preds)
        label_preds = np.where(label_preds==class_no, nc_class, label_preds)
    if not shortened:
        dice_per_class = dice_coef_multilabel(label_trues, label_preds)
        results['macro_f1'] = f1_score(label_trues, label_preds, labels=np.arange(class_no), average='macro', zero_division=0)
        f1_score_classwise = f1_score(label_trues, label_preds, labels=np.arange(class_no), average=None, zero_division=0)

        results['macro_dice'] = dice_per_class.mean()

        intersection = (label_preds == label_trues).sum(axis=None)
        sum_ = 2 * np.prod(label_preds.shape)
        results['micro_dice'] = ((2 * intersection + 1e-6) / (sum_ + 1e-6))

        for class_id in range(class_no):
            # results['dice_class_' + str(class_id) + '_' + class_names[class_id]] = dice_per_class[class_id]
            results['f1_class_' + str(class_id) + '_' + class_names[class_id]] = f1_score_classwise[class_id]

    results['accuracy'] = accuracy_score(label_trues, label_preds)
    results['miou'] = jaccard_score(label_trues, label_preds, average="macro") # same as IoU!
    results['cohens_kappa'] = cohen_kappa_score(label_trues, label_preds, weights=None)
    results['cohens_kappa_quad'] = cohen_kappa_score(label_trues, label_preds, weights='quadratic')

    return results