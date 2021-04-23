import glob

import numpy as np
import skimage.io
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
    _, m, n = im.shape
    mask_x = np.arange(0, n, stride)
    mask_y = np.arange(0, m, stride)
    mask[:, :, mask_x] = 1
    mask[:, mask_y, :] = 1
        
    return (im * mask).type(torch.float32)


def load_images(path='data/healthy_small'):
    """Load all jpg images in a folder."""
    
    glob_path = path + '/*.jpg'
    ims = []
    
    for i, file in enumerate(glob.glob(glob_path)):
        im = skimage.io.imread(file)
        im = im.astype(np.float32) / 255
        ims.append(im)
        
    ims = np.array(ims).astype(np.float32)
    
    return ims