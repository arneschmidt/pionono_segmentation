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
            classes=class_no # model output channels (number of classes in your dataset)
        )
    else:
        raise Exception('Choose valid model backbone!')
    return seg_model

class SegmentationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seg_model = create_segmentation_backbone()
        self.activation = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.seg_model(x)
        y = self.activation(x)
        return y


class cm_layers(nn.Module):
    """ This class defines the annotator network, which models the confusion matrix.
    Essentially, it share the semantic features with the segmentation network, but the output of annotator network
    has the size (b, c**2, h, w)
    """

    def __init__(self, in_channels, norm, class_no):
        super(cm_layers, self).__init__()
        self.conv_1 = double_conv(in_channels=in_channels, out_channels=in_channels, norm=norm, step=1)
        self.conv_2 = double_conv(in_channels=in_channels, out_channels=in_channels, norm=norm, step=1)
        self.conv_last = nn.Conv2d(in_channels, class_no**2, 1, bias=True)
        self.relu = nn.Softplus()

    def forward(self, x):

        y = self.relu(self.conv_last(self.conv_2(self.conv_1(x))))

        return y

class Crowd_segmentationModel(torch.nn.Module):
    def __init__(self, no_noisy_labels):
        super().__init__()
        self.seg_model = create_segmentation_backbone()
        self.activation = torch.nn.Softmax(dim=1)
        self.no_noisy_labels = no_noisy_labels
        print("Number of annotators (model): ", self.no_noisy_labels)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_last = nn.Conv2d(width, self.final_in, 1, bias=True)
        self.spatial_cms = []
        for i in range(self.noisy_labels_no):
            self.spatial_cms.append(cm_layers(in_channels=3, norm='in', class_no=config['data']['class_no']))
        self.activation = torch.nn.Softmax(dim=1)
    def forward(self, x):
        cms = []
        x = self.seg_model.encoder(x)
        x = self.seg_model.decoder(x)
        for i in range(self.noisy_labels_no):
            cm = self.spatial_cms[i](x)
            cms.append(cm)
        x = self.seg_model.segmentation_head(x)
        y = self.activation(x)
        return y, cms
