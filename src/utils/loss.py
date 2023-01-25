import torch
import utils.globals as globals
import torch.nn as nn
import torch.nn.functional as F
torch.backends.cudnn.deterministic = True
# =======================================

from segmentation_models_pytorch.losses import DiceLoss, FocalLoss

eps=1e-7

def noisy_label_loss(pred, cms, labels, loss_fct, min_trace = False, alpha=0.1):
    """ This function defines the proposed trace regularised loss function, suitable for either binary
    or multi-class segmentation task. Essentially, each pixel has a confusion matrix.
    Args:
        pred (torch.tensor): output tensor of the last layer of the segmentation network without Sigmoid or Softmax
        cms (list): a list of output tensors for each noisy label, each item contains all of the modelled confusion matrix for each spatial location
        labels (torch.tensor): labels
        alpha (double): a hyper-parameter to decide the strength of regularisation
    Returns:
        loss (double): total loss value, sum between main_loss and regularisation
        main_loss (double): main segmentation loss
        regularisation (double): regularisation loss
    """
    b, c, h, w = pred.size()
    if globals.config['data']['ignore_last_class']:
        ignore_index = int(globals.config['data']['class_no'])  # deleted class is always set to the last index
    else:
        ignore_index = -100
    # normalise the segmentation output tensor along dimension 1
    pred_norm = pred

    # b x c x h x w ---> b*h*w x c x 1
    pred_norm = pred_norm.view(b, c, h*w).permute(0, 2, 1).contiguous().view(b*h*w, c, 1)


    # cm: learnt confusion matrix for each noisy label, b x c**2 x h x w
    # label_noisy: noisy label, b x h x w

    # b x c**2 x h x w ---> b*h*w x c x c
    cm = cms.view(b, c ** 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, c * c).view(b * h * w, c, c)
    cm = cm / cm.sum(1, keepdim=True) # normalization
    # matrix multiplication to calculate the predicted noisy segmentation:
    # cm: b*h*w x c x c
    # pred_noisy: b*h*w x c x 1
    # print(cm.shape, pred_norm.shape)
    pred_noisy = torch.bmm(cm, pred_norm).view(b*h*w, c)
    pred_noisy = pred_noisy.view(b, h*w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)
    log_likelihood_loss = loss_fct(pred_noisy, labels.view(b, h, w).long())
    # regularization
    regularisation = torch.trace(torch.transpose(torch.sum(cm, dim=0), 0, 1)).sum() / (b * h * w)
    regularisation = alpha * regularisation

    if min_trace:
        loss = log_likelihood_loss + regularisation
    else:
        loss = log_likelihood_loss - regularisation

    return loss, log_likelihood_loss, regularisation
