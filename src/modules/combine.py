import torch
import torch.nn as nn

class Combine(nn.Module):
    def __init__(self, mode='and'):
        super().__init__()
        self.mode = mode

    def forward(self, attn1, attn2):
        if self.mode == 'and':
            return attn1 * attn2
        elif self.mode == 'except':
            return attn1 * (1 - attn2)
        else:
            raise ValueError("Unknown combine mode")
