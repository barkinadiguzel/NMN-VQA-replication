import torch
import torch.nn as nn

def vqa_loss(pred, target):
    criterion = nn.CrossEntropyLoss()
    return criterion(pred, target)
