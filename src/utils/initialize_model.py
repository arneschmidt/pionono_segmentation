import torch
import utils.globals as globals
from utils.model_supervised import SupervisedSegmentationModel
from utils.model_confusionmatrix import ConfusionMatrixModel
from Probabilistic_Unet_Pytorch.probabilistic_unet import ProbabilisticUnet
from utils.model_pionono import PiononoModel

def init_model(annotators):
    config = globals.config

    if config['model']['method'] == 'prob_unet':
        model = ProbabilisticUnet(input_channels=3, num_classes=config['data']['class_no'],
                                       latent_dim=config['model']['prob_unet_config']['latent_dim'],
                                       no_convs_fcomb=4, beta=config['model']['prob_unet_config']['kl_factor'],
                                       reg_factor=config['model']['prob_unet_config']['reg_factor'],
                                       original_backbone=config['model']['prob_unet_config']['original_backbone'])
        # (256, 128, 64, 32, 16)
        # model = ProbabilisticUnet(3, config['data']['class_no'])
    elif config['model']['method'] == 'pionono':
        model = PiononoModel(input_channels=3, num_classes=config['data']['class_no'],
                             annotators=annotators,
                             gold_annotators=config['model']['pionono_config']['gold_annotators'],
                             latent_dim=config['model']['pionono_config']['latent_dim'],
                             no_head_layers=config['model']['pionono_config']['no_head_layers'],
                             head_kernelsize=config['model']['pionono_config']['head_kernelsize'],
                             head_dilation=config['model']['pionono_config']['head_dilation'],
                             kl_factor=config['model']['pionono_config']['kl_factor'],
                             reg_factor=config['model']['pionono_config']['reg_factor'],
                             mc_samples=config['model']['pionono_config']['mc_samples'],
                             z_prior_sigma=config['model']['pionono_config']['z_prior_sigma'],
                             z_posterior_init_sigma=config['model']['pionono_config']['z_posterior_init_sigma'],
                             )
    elif config['model']['method'] == 'conf_matrix':
        model = ConfusionMatrixModel(num_classes=config['data']['class_no'], num_annotators=len(annotators),
                                     level=config['model']['conf_matrix_config']['level'],
                                     image_res=config['data']['image_resolution'],
                                     learning_rate=config['model']['learning_rate'],
                                     alpha=config['model']['conf_matrix_config']['alpha'],
                                     min_trace=config['model']['conf_matrix_config']['min_trace'])
    else:
        model = SupervisedSegmentationModel()

    return model