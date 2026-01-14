import torch
import torch.nn as nn
import torch.nn.functional as F

class ReAttend(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, attn):
        B, C, H, W = attn.shape
        x = attn.view(B, C, -1).mean(-1)  
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.unsqueeze(-1).unsqueeze(-1).expand(B, C, H, W)
        return x
