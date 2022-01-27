from src.utils.globals import config
import segmentation_models_pytorch as smp


def create_model():
    model = smp.Unet(
        encoder_name=config['model']['encoder'],  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=config['model']['imagenet'],  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=config['data']['class_no'],  # model output channels (number of classes in your dataset)
        activation=config['model']['activation']
    )

    model.cuda()  # to(device)

    return model