import torch
import torch.nn as nn
import torchvision.models as models

class CNNEncoder(nn.Module):
    def __init__(self, output_dim=512, pretrained=True):
        super().__init__()
        resnet = models.resnet18(pretrained=pretrained)
        modules = list(resnet.children())[:-2]  
        self.backbone = nn.Sequential(*modules)
        self.output_dim = output_dim
        self.conv = nn.Conv2d(512, output_dim, kernel_size=1)

    def forward(self, x):
        features = self.backbone(x)
        features = self.conv(features)
        return features  
