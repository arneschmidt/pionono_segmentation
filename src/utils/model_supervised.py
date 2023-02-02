import torch
from utils.segmentation_backbone import create_segmentation_backbone

class SupervisedSegmentationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seg_model = create_segmentation_backbone()
        self.activation = torch.nn.Softmax(dim=1)

    def forward(self, x, use_softmax=False):
        x = self.seg_model(x)
        y = self.activation(x)
        return y

    def train_step(self, images, labels, loss_fct, ann_ids):
        y_pred = self.forward(images)
        loss = loss_fct(y_pred, labels)
        return loss, y_pred
