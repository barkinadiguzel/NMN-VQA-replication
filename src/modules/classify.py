import torch
import torch.nn as nn
import torch.nn.functional as F

class Classify(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, image_features, attn):
        B, C, H, W = image_features.shape
        attn = F.softmax(attn.view(B, C, -1), dim=-1).view(B, C, H, W)
        pooled = (image_features * attn).view(B, C, -1).sum(-1)
        out = self.fc(pooled)
        return out
