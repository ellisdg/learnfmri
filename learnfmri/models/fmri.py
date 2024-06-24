# for now, the fMRI model will be a 3D resnet classifier network
from monai.networks.nets import resnet50
import torch


class FMRIResNet3D(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, spatial_dims: int = 3):
        super(FMRIResNet3D, self).__init__()
        self.resnet = resnet50(spatial_dims=spatial_dims, in_channels=in_channels, num_classes=out_channels)

    def forward(self, x):
        return self.resnet(x)


