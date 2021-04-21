import torch
import torch.nn as nn

def orthogonal_init(module, gain=nn.init.calculate_gain('relu')):
    """Orthogonal weight initialization: https://arxiv.org/abs/1312.6120"""
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.orthogonal_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module