import torch
from utils.segmentation_backbone import create_segmentation_backbone

class SupervisedSegmentationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seg_model = create_segmentation_backbone()
        self.activation = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.seg_model(x)
        y = self.activation(x)
        return y