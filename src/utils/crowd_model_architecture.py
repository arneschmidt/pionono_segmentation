import torch
from utils.model_architecture import create_segmentation_backbone
import utils.globals

def double_conv(in_channels, out_channels, step, norm):
    # ===========================================
    # in_channels: dimension of input
    # out_channels: dimension of output
    # step: stride
    # ===========================================
    if norm == 'in':
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, stride=step, padding=1, groups=1, bias=False),
            torch.nn.InstanceNorm2d(out_channels, affine=True),
            torch.nn.PReLU(),
            torch.nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, groups=1, bias=False),
            torch.nn.InstanceNorm2d(out_channels, affine=True),
            torch.nn.PReLU()
        )
    elif norm == 'bn':
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, stride=step, padding=1, groups=1, bias=False),
            torch.nn.BatchNorm2d(out_channels, affine=True),
            torch.nn.PReLU(),
            torch.nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, groups=1, bias=False),
            torch.nn.BatchNorm2d(out_channels, affine=True),
            torch.nn.PReLU()
        )
    elif norm == 'ln':
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, stride=step, padding=1, groups=1, bias=False),
            torch.nn.GroupNorm(out_channels, out_channels, affine=True),
            torch.nn.PReLU(),
            torch.nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, groups=1, bias=False),
            torch.nn.GroupNorm(out_channels, out_channels, affine=True),
            torch.nn.PReLU()
        )
    elif norm == 'gn':
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, stride=step, padding=1, groups=1, bias=False),
            torch.nn.GroupNorm(out_channels // 8, out_channels, affine=True),
            torch.nn.PReLU(),
            torch.nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, groups=1, bias=False),
            torch.nn.GroupNorm(out_channels // 8, out_channels, affine=True),
            torch.nn.PReLU()
        )


class cm_layers(torch.nn.Module):
    """ This class defines the annotator network, which models the confusion matrix.
    Essentially, it share the semantic features with the segmentation network, but the output of annotator network
    has the size (b, c**2, h, w)
    """

    def __init__(self, in_channels, norm, class_no):
        super(cm_layers, self).__init__()
        self.conv_1 = double_conv(in_channels=in_channels, out_channels=in_channels, norm=norm, step=1)
        self.conv_2 = double_conv(in_channels=in_channels, out_channels=in_channels, norm=norm, step=1)
        self.conv_last = torch.nn.Conv2d(in_channels, class_no ** 2, 1, bias=True)
        self.relu = torch.nn.Softplus()

    def forward(self, x):
        y = self.relu(self.conv_last(self.conv_2(self.conv_1(x))))

        return y


class Crowd_segmentationModel(torch.nn.Module):
    def __init__(self, noisy_labels_no):
        super().__init__()
        config = utils.globals.config
        self.seg_model = create_segmentation_backbone()
        self.activation = torch.nn.Softmax(dim=1)
        self.noisy_labels_no = noisy_labels_no
        print("Number of annotators (model): ", self.noisy_labels_no)
        self.spatial_cms = torch.nn.ModuleList()
        for i in range(self.noisy_labels_no):
            self.spatial_cms.append(cm_layers(in_channels=16, norm='in', class_no=config['data']['class_no'])) # TODO: arrange in_channels
        self.activation = torch.nn.Softmax(dim=1)

    def forward(self, x):
        cms = []
        x = self.seg_model.encoder(x)
        x = self.seg_model.decoder(*x)
        for i in range(self.noisy_labels_no):
            cm = self.spatial_cms[i](x)
            cms.append(cm)
        x = self.seg_model.segmentation_head(x)
        y = self.activation(x)
        return y, cms
