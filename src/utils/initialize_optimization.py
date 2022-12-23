import torch
import utils.globals as globals
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss

def init_optimization(model):
    config = globals.config
    learning_rate = config['model']['learning_rate']
    c_weights = config['data']['class_weights']
    class_weights = torch.FloatTensor(c_weights).cuda()

    if config['model']['method'] == 'pionono':
        opt_params = [
            {'params': model.unet.parameters()},
            {'params': model.head.parameters()},
            {'params': model.z.parameters(), 'lr': config['model']['pionono_config']['z_learning_rate']}
        ]
    elif config['model']['method'] == 'confusion_matrix':
        opt_params = [
            {'params': model.seg_model.parameters()},
            {'params': model.crowd_layers.parameters(), 'lr': 1e-3}
        ]
    else:
        opt_params = [{'params': model.parameters()}]

    if config['model']['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(opt_params, lr=learning_rate)
    elif config['model']['optimizer'] == 'sgd_mom':
        optimizer = torch.optim.SGD(opt_params, lr=learning_rate, momentum=0.9, nesterov=True)
    else:
        raise Exception('Choose valid optimizer!')

    if config['data']['ignore_last_class']:
        ignore_index = int(config['data']['class_no'])  # deleted class is always set to the last index
    else:
        ignore_index = -100  # this means no index ignored

    loss_mode = config['model']['loss']
    if config['model']['method'] != 'conf_matrix':
        if loss_mode == 'ce':
            loss_fct = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=ignore_index, weight=class_weights)
        elif loss_mode == 'dice':
            loss_fct = DiceLoss(ignore_index=ignore_index, from_logits=False, mode='multiclass')
        elif loss_mode == 'focal':
            loss_fct = FocalLoss(reduction='mean', ignore_index=ignore_index, mode='multiclass')
        else:
            raise Exception('Choose valid loss function!')

    return optimizer, loss_fct