import torch
import utils.globals
import segmentation_models_pytorch as smp


def create_segmentation_backbone():
    config = utils.globals.config
    class_no = config['data']['class_no']

    if config['model']['backbone'] == 'unet':
        seg_model = smp.Unet(
            encoder_name=config['model']['encoder']['backbone'],  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=config['model']['encoder']['weights'],
            # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=class_no  # model output channels (number of classes in your dataset)
        )
    elif config['model']['backbone'] == 'linknet':
        seg_model = smp.Linknet(
            encoder_name=config['model']['encoder']['backbone'],  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=config['model']['encoder']['weights'],
            # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=class_no  # model output channels (number of classes in your dataset)
        )
    else:
        raise Exception('Choose valid model backbone!')
    return seg_model

