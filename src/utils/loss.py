import torch
import torch.nn as nn
import torch.nn.functional as F
torch.backends.cudnn.deterministic = True
# =======================================


def noisy_label_loss(pred, cms, labels, ignore_index, alpha=0.1):
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
    main_loss = 0.0
    regularisation = 0.0
    b, c, h, w = pred.size()

    # normalise the segmentation output tensor along dimension 1
    pred_norm = pred

    # b x c x h x w ---> b*h*w x c x 1
    pred_norm = pred_norm.view(b, c, h*w).permute(0, 2, 1).contiguous().view(b*h*w, c, 1)

    for cm, label_noisy in zip(cms, labels):
        # cm: learnt confusion matrix for each noisy label, b x c**2 x h x w
        # label_noisy: noisy label, b x h x w

        # b x c**2 x h x w ---> b*h*w x c x c
        cm = cm.view(b, c ** 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, c * c).view(b * h * w, c, c)

        # normalisation along the rows:
        # print(cm.shape)
        cm = cm / cm.sum(1, keepdim=True)
        print(cm[0])

        # matrix multiplication to calculate the predicted noisy segmentation:
        # cm: b*h*w x c x c
        # pred_noisy: b*h*w x c x 1
        pred_noisy = torch.bmm(cm, pred_norm).view(b*h*w, c)
        pred_noisy = pred_noisy.view(b, h*w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)
        loss_current = nn.CrossEntropyLoss(reduction='mean', ignore_index=ignore_index)(pred_noisy, label_noisy.view(b, h, w).long())
        # print("loss_current: ", loss_current)
        main_loss += loss_current
        print("annotator loss: ", loss_current)
        regularisation += torch.trace(torch.transpose(torch.sum(cm, dim=0), 0, 1)).sum() / (b * h * w)

    regularisation = alpha*regularisation
    main_loss = main_loss/3
    loss = main_loss + regularisation
    print("main loss: ", main_loss, "\n Regularization: ", regularisation)

    return loss, main_loss, regularisation