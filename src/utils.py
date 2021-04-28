import glob

import numpy as np
import skimage.io
from scipy import interpolate

import torch
import torch.nn as nn

def orthogonal_init(module, gain=nn.init.calculate_gain('relu')):
    """Orthogonal weight initialization: https://arxiv.org/abs/1312.6120"""
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.orthogonal_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module


def down_sample(im, stride=4, is_torch=True):
    """Downsampling corresponding to reducing number of linescans in x- and y-direction."""
    mask = np.zeros(im.shape)
    
    # Handle torch and numpy different.
    if is_torch:
        _, m, n = im.shape
    else:
        m, n, _ = im.shape

    mask_x = np.arange(0, n, stride)
    mask_y = np.arange(0, m, stride)
    
    if is_torch:
        mask[:, :, mask_x] = 1
        mask[:, mask_y, :] = 1
        
        ds = (im * mask).type(torch.float32)
    else:
        mask[:, mask_x] = 1
        mask[mask_y, :] = 1
        
        ds = (im * mask).astype(np.float32)
        
    return ds, mask


def interpolate_rgb(im, mask2d):
    """Interpolate RGB data by interpolating griddata for each channel."""
    im_interpolated = np.zeros(im.shape)

    x = np.arange(0, im.shape[1])
    y = np.arange(0, im.shape[0])
    xx, yy = np.meshgrid(x, y)

    x1 = xx[~mask2d]
    y1 = yy[~mask2d]
    
    for ch in range(im.shape[2]):
        new_im = im[~mask2d, ch]

        im_interpolated[..., ch] = interpolate.griddata((x1, y1), new_im.ravel(), 
                                                        (xx, yy), method='nearest')
    return im_interpolated.astype(np.float32)


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