import torch
import torch.nn as nn
import torch.nn.functional as F

class Measure(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, attn):
        B, C, H, W = attn.shape
        x = attn.view(B, C, -1).sum(-1)  
        x = F.relu(self.fc1(x))
        out = self.fc2(x)
        return out
