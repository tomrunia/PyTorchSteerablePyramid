# MIT License
#
# Copyright (c) 2018 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-12-10

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import skimage
import matplotlib.pyplot as plt

import torch
import torchvision

################################################################################

ToPIL = torchvision.transforms.ToPILImage()
Grayscale = torchvision.transforms.Grayscale()
RandomCrop = torchvision.transforms.RandomCrop

def get_device(device='cuda:0'):
    assert isinstance(device, str)
    num_cuda = torch.cuda.device_count()

    if 'cuda' in device:
        if num_cuda > 0:
            # Found CUDA device, use the GPU
            return torch.device(device)
        # Fallback to CPU
        print('No CUDA devices found, falling back to CPU')
        device = 'cpu'

    if not torch.backends.mkl.is_available():
        raise NotImplementedError(
            'torch.fft on the CPU requires MKL back-end. ' +
            'Please recompile your PyTorch distribution.')
    return torch.device('cpu')

def load_image_batch(image_file, batch_size, image_size=200):
    if not os.path.isfile(image_file):
        raise FileNotFoundError('Image file not found on disk: {}'.format(image_file))
    im = ToPIL(skimage.io.imread(image_file))
    im = Grayscale(im)
    im_batch = np.zeros((batch_size, image_size, image_size), np.float32)
    for i in range(batch_size):
        im_batch[i] = RandomCrop(image_size)(im)
    # insert channels dim and rescale
    return im_batch[:,None,:,:]/225.

def show_image_batch(im_batch):
    assert isinstance(im_batch, torch.Tensor)
    im_batch = torchvision.utils.make_grid(im_batch).numpy()
    im_batch = np.transpose(im_batch.squeeze(1), (1,2,0))
    plt.imshow(im_batch)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    return im_batch

def extract_from_batch(coeff_batch, example_idx=0):
    '''
    Given the batched Complex Steerable Pyramid, extract the coefficients
    for a single example from the batch. Additionally, it converts all
    torch.Tensor's to np.ndarrays' and changes creates proper np.complex
    objects for all the orientation bands. 

    Args:
        coeff_batch (list): list containing low-pass, high-pass and pyr levels
        example_idx (int, optional): Defaults to 0. index in batch to extract
    
    Returns:
        list: list containing low-pass, high-pass and pyr levels as np.ndarray
    '''
    if not isinstance(coeff_batch, list):
        raise ValueError('Batch of coefficients must be a list')
    coeff = []  # coefficient for single example
    for coeff_level in coeff_batch:
        if isinstance(coeff_level, torch.Tensor):
            # Low- or High-Pass
            coeff_level_numpy = coeff_level[example_idx].cpu().numpy()
            coeff.append(coeff_level_numpy)
        elif isinstance(coeff_level, list):
            coeff_orientations_numpy = []
            for coeff_orientation in coeff_level:
                coeff_orientation_numpy = coeff_orientation[example_idx].cpu().numpy()
                coeff_orientation_numpy = coeff_orientation_numpy[:,:,0] + 1j*coeff_orientation_numpy[:,:,1]
                coeff_orientations_numpy.append(coeff_orientation_numpy)
            coeff.append(coeff_orientations_numpy)
        else:
            raise ValueError('coeff leve must be of type (list, torch.Tensor)')
    return coeff

################################################################################

def make_grid_coeff(coeff, normalize=True):
    '''
    Visualization function for building a large image that contains the
    low-pass, high-pass and all intermediate levels in the steerable pyramid. 
    For the complex intermediate bands, the real part is visualized.
    
    Args:
        coeff (list): complex pyramid stored as list containing all levels
        normalize (bool, optional): Defaults to True. Whether to normalize each band
    
    Returns:
        np.ndarray: large image that contains grid of all bands and orientations
    '''
    M, N = coeff[1][0].shape
    Norients = len(coeff[1])
    out = np.zeros((M * 2 - coeff[-1].shape[0], Norients * N))
    currentx, currenty = 0, 0

    for i in range(1, len(coeff[:-1])):
        for j in range(len(coeff[1])):
            tmp = coeff[i][j].real
            m, n = tmp.shape
            if normalize:
                tmp = 255 * tmp/tmp.max()
            tmp[m-1,:] = 255
            tmp[:,n-1] = 255
            out[currentx:currentx+m,currenty:currenty+n] = tmp
            currenty += n
        currentx += coeff[i][0].shape[0]
        currenty = 0

    m, n = coeff[-1].shape
    out[currentx: currentx+m, currenty: currenty+n] = 255 * coeff[-1]/coeff[-1].max()
    out[0,:] = 255
    out[:,0] = 255
    return out
