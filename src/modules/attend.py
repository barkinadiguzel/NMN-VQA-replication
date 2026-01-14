import torch
import torch.nn as nn
import torch.nn.functional as F

class Attend(nn.Module):
    def __init__(self, input_dim, attn_dim=128):
        super().__init__()
        self.conv = nn.Conv2d(input_dim, attn_dim, kernel_size=1)

    def forward(self, image_features):
        attn_map = self.conv(image_features)  
        attn_map = F.relu(attn_map)
        return attn_map
