# for now, the t1w model will be a 3D resnet classifier network
import torch
from monai.networks.nets import resnet50


class T1WResNet3D(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, spatial_dims: int = 3):
        super(T1WResNet3D, self).__init__()
        self.resnet = resnet50(spatial_dims=spatial_dims, in_channels=in_channels, num_classes=out_channels)

    def forward(self, x):
        return self.resnet(x)
