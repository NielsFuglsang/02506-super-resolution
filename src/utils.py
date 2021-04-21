import numpy as np

import torch
import torch.nn as nn

def orthogonal_init(module, gain=nn.init.calculate_gain('relu')):
    """Orthogonal weight initialization: https://arxiv.org/abs/1312.6120"""
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.orthogonal_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module


def down_sample(im, stride=4):
    """Downsampling corresponding to reducing number of linescans in x- and y-direction."""
    mask = np.zeros(im.shape)
    m, n, _ = im.shape
    mask_x = np.arange(0, n, stride)
    mask_y = np.arange(0, m, stride)
    mask[:, mask_x] = 1
    mask[mask_y, :] = 1
    
    # Information kept.
    # print(np.mean(mask))
    
    return (im * mask).astype(np.float32)