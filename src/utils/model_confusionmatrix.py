import numpy as np
import torch
import torch.nn.functional as F
from utils.segmentation_backbone import create_segmentation_backbone
import utils.globals as globals
from utils.loss import noisy_label_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


class gcm_layers(torch.nn.Module):
    """ This defines the global confusion matrix layer. It defines a (class_no x class_no) confusion matrix, we then use unsqueeze function to match the
    size with the original pixel-wise confusion matrix layer, this is due to convenience to be compact with the existing loss function and pipeline.
    """

    def __init__(self, class_no, input_height, input_width, noisy_labels_no):
        super(gcm_layers, self).__init__()
        self.class_no = class_no
        self.noisy_labels_no = noisy_labels_no
        self.input_height = input_height
        self.input_width = input_width
        self.global_weights = torch.nn.Parameter(torch.eye(class_no))
        self.dense_annotator = torch.nn.Linear(self.noisy_labels_no, torch.ceil(self.noisy_labels_no / 2))
        self.relu = torch.nn.ReLU()

    def forward(self, x, A_id):
        A_feat = self.dense_annotator(A_id)
        A_feat = self.relu(A_feat)
        torch.nn.torch.eye(self.class_no)
        all_weights = self.global_weights.unsqueeze(0).repeat(x.size(0), 1, 1)
        all_weights = all_weights.unsqueeze(3).unsqueeze(4).repeat(1, 1, 1, self.input_height, self.input_width)
        # y = self.relu(all_weights)
        y = all_weights

        return y


class cm_layers(torch.nn.Module):
    """ This class defines the annotator network, which models the confusion matrix.
    Essentially, it share the semantic features with the segmentation network, but the output of annotator network
    has the size (b, c**2, h, w)
    """

    def __init__(self, in_channels, input_height, input_width, norm, class_no, noisy_labels_no):
        super(cm_layers, self).__init__()
        self.input_height = input_height
        self.input_width = input_width
        self.conv_1 = double_conv(in_channels=in_channels, out_channels=in_channels, norm=norm, step=1)
        self.conv_2 = double_conv(in_channels=in_channels, out_channels=in_channels, norm=norm, step=1)
        # self.conv_last = torch.nn.Conv2d(in_channels, class_no ** 2, 1, bias=True)
        self.class_no = class_no
        self.dense = torch.nn.Linear(80, 25)
        self.dense2 = torch.nn.Linear(25, 25)
        self.dense_annotator = torch.nn.Linear(noisy_labels_no, 64)
        # self.dense_classes = torch.nn.Linear(noisy_labels_no, 50)
        self.norm = torch.nn.BatchNorm2d(80, affine=True)
        self.relu = torch.nn.Softplus()
        self.act = torch.nn.Softmax(dim=3)

    def forward(self, A_id, x):
        # TODO add the one-hot to the x
        # A_id 20
        # x BxWxHx16 -> BxWxHx16+20
        #y = self.relu(self.conv_last(self.conv_2(self.conv_1(x))))
        y = self.conv_2(self.conv_1(x))
        A_id = self.relu(self.dense_annotator(A_id))  # B, F_A
        #print("A_id", A_id.shape)
        #print("image features", y.shape)
        A_id = A_id.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.input_height, self.input_width)
        #print("A_id", A_id.shape)

        y = torch.cat((A_id, y), dim=1)
        #print(y.shape)
        y = self.norm(y)
        #print("y", y.shape)
        y = y.permute(0, 2, 3, 1)
        y = self.relu((self.dense(y)))
        y = self.dense2(y)
        
        y = self.relu(y.view(-1, self.input_height, self.input_width, self.class_no, self.class_no))
        y = y.view(-1, self.input_height, self.input_width, self.class_no ** 2).permute(0,3,1,2)

        return y


class global_CM(torch.nn.Module):
    """ This defines the global confusion matrix layer. It defines a (class_no x class_no) confusion matrix, we then use unsqueeze function to match the
    size with the original pixel-wise confusion matrix layer, this is due to convenience to be compact with the existing loss function and pipeline.
    """

    def __init__(self, class_no, input_height, input_width, noisy_labels_no):
        super(global_CM, self).__init__()
        self.class_no = class_no
        self.noisy_labels_no = noisy_labels_no
        self.input_height = input_height
        self.input_width = input_width
        self.noisy_labels_no = noisy_labels_no
        self.dense_output = torch.nn.Linear(noisy_labels_no, class_no ** 2)
        self.act = torch.nn.Softplus()
        # self.relu = torch.nn.ReLU()

    def forward(self, A_id, x=None):
        #A_feat = self.relu(self.dense_annotator(A_id))  # B, F_A
        #feat_class = self.relu(self.dense_classes(A_feat))
        #output = self.dense_output(feat_class)
        #output = self.act(output.view(-1, self.class_no, self.class_no))
        output = self.act(self.dense_output(A_id))
        all_weights = output.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, self.input_height, self.input_width)
        #print(all_weights.shape)
        #y = all_weights / all_weights.sum(1, keepdim=True)
        # print(y.sum(1))
        #print(y.shape)
        y = all_weights.view(-1, self.class_no**2, self.input_height, self.input_width)



        return y


class conv_layers_image(torch.nn.Module):
    def __init__(self, in_channels):
        super(conv_layers_image, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_bn = torch.nn.BatchNorm2d(8)
        self.conv_bn2 = torch.nn.BatchNorm2d(4)
        self.fc_bn = torch.nn.BatchNorm1d(128)
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(in_features=16384, out_features=128)
        self.fc2 = torch.nn.Linear(in_features=128, out_features=64)


    def forward(self, x):
        x = self.pool(self.relu(self.conv_bn(self.conv(x))))
        x = self.pool(self.relu(self.conv_bn2(self.conv2(x))))
        x = self.pool(self.relu(self.conv_bn2(self.conv3(x))))
        x = self.pool(self.relu(self.conv_bn2(self.conv3(x))))
        x = self.flatten(x)

        x = self.relu(self.fc_bn(self.fc1(x)))
        y = self.fc2(x)

        return y


class image_CM(torch.nn.Module):
    """ This defines the global confusion matrix layer. It defines a (class_no x class_no) confusion matrix, we then use unsqueeze function to match the
    size with the original pixel-wise confusion matrix layer, this is due to convenience to be compact with the existing loss function and pipeline.
    """

    def __init__(self, class_no, input_height, input_width, noisy_labels_no):
        super(image_CM, self).__init__()
        self.class_no = class_no
        self.noisy_labels_no = noisy_labels_no
        self.input_height = input_height
        self.input_width = input_width
        self.noisy_labels_no = noisy_labels_no
        self.conv_layers = conv_layers_image(16)
        self.dense_annotator = torch.nn.Linear(noisy_labels_no, 64)
        self.dense_output = torch.nn.Linear(128, class_no ** 2)
        self.norm = torch.nn.BatchNorm1d(class_no ** 2)
        self.act = torch.nn.Softplus()

    def forward(self, A_id, x):
        A_feat = self.dense_annotator(A_id)  # B, F_A
        x = self.conv_layers(x)
        output = self.dense_output(torch.hstack((A_feat, x)))
        output = self.norm(output)
        output = self.act(output.view(-1, self.class_no, self.class_no))
        all_weights = output.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, self.input_height, self.input_width)
        y = all_weights.view(-1, self.class_no**2, self.input_height, self.input_width)
        #print("shape CM image ", y.shape)

        return y


class ConfusionMatrixModel(torch.nn.Module):
    def __init__(self, num_classes, annotators, level, image_res, learning_rate, alpha, min_trace):
        super().__init__()
        config = globals.config
        self.seg_model = create_segmentation_backbone().to(device)
        self.num_annotators = len(annotators)
        self.num_classes = num_classes
        self.level = level
        self.image_res = image_res
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.min_trace = min_trace
        if self.level == 'global':
            print("Global crowdsourcing")
            self.crowd_layers = global_CM(self.num_classes, self.image_res, self.image_res, self.num_annotators)
        elif self.level == 'image':
            print("Image dependent crowdsourcing")
            self.crowd_layers = image_CM(self.num_classes, self.image_res, self.image_res, self.num_annotators)
        elif self.level == 'pixel':
            print("Pixel dependent crowdsourcing")
            self.crowd_layers = cm_layers(in_channels=16, input_height=image_res, input_width=image_res, norm='in',
                                          class_no=self.num_classes, noisy_labels_no=self.num_annotators)  # TODO: arrange in_channels
        self.crowd_layers.to(device)
        self.activation = torch.nn.Softmax(dim=1)

    def forward(self, x, A_id=None):
        cm = None
        x = self.seg_model.encoder(x)
        x = self.seg_model.decoder(*x)
        if A_id is not None:
            A_onehot = F.one_hot(A_id.long(), self.num_annotators)
            cm = self.crowd_layers(A_onehot.float(), x)
        x = self.seg_model.segmentation_head(x)
        y = self.activation(x)
        return y, cm

    def train_step(self, images, labels, loss_fct, ann_ids):
        y_pred, cms = self.forward(images, ann_ids)
        loss, loss_ce, loss_trace = noisy_label_loss(y_pred, cms, labels, loss_fct,
                                                     self.min_trace,
                                                     self.alpha)
        return loss, y_pred

    def activate_min_trace(self):
        print("Minimize trace activated!")
        min_trace = True
        print("Alpha updated", self.alpha)
        optimizer = torch.optim.Adam([
            {'params': self.seg_model.parameters()},
            {'params': self.crowd_layers.parameters(), 'lr': 1e-4}
        ], lr=self.learning_rate)
        return optimizer
