import src.utils.globals
import segmentation_models_pytorch as smp

def create_model():
    config = src.utils.globals.config

    if config['model']['decoder']['activation'] == 'None':
        activation = None
    else:
        activation = config['model']['decoder']['activation']

    if config['model']['backbone'] == 'unet':
        model = smp.Unet(
            encoder_name=config['model']['encoder']['backbone'],  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=config['model']['encoder']['weights'],  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=config['data']['class_no'],  # model output channels (number of classes in your dataset)
            activation=activation
        )
    elif config['model']['backbone'] == 'linknet':
        model = smp.Linknet(
            encoder_name=config['model']['encoder']['backbone'],  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=config['model']['encoder']['weights'],  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=config['data']['class_no'],  # model output channels (number of classes in your dataset)
            activation=activation
        )
    else:
        raise Exception('Choose valid model backbone!')

    model.cuda()  # to(device)

    return model