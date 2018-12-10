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
        else:
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
    return im_batch/225.

def show_image_batch(im_batch):
    assert isinstance(im_batch, torch.Tensor)
    im_batch = torchvision.utils.make_grid(im_batch).numpy()
    im_batch = np.transpose(im_batch, (1,2,0))
    plt.imshow(im_batch)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    return im_batch
        
################################################################################

if __name__ == "__main__":
    filename = './assets/patagonia.jpg'
    im_batch = load_image_batch(filename, 16, 200)
    im_batch = torch.from_numpy(im_batch[:,None,:,:])
    show_image_batch(im_batch)