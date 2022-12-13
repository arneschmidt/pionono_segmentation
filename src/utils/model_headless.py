import torch
from utils.segmentation_backbone import create_segmentation_backbone


class UnetHeadless(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seg_model = create_segmentation_backbone()

    def forward(self, x):
        x = self.seg_model.encoder(x)
        x = self.seg_model.decoder(*x)
        # x = self.seg_model(x)
        return x