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
# Date Created: 2018-12-04

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import cv2

from steerable.fft import fftshift

################################################################################

image_file = './assets/lena.jpg'
im = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
im = cv2.resize(im, dsize=(200, 200))
im = im.astype(np.float64)/255.  # note the np.float64

################################################################################
# NumPy

fft_numpy = np.fft.fft2(im)
fft_numpy = np.fft.fftshift(fft_numpy)

fft_numpy_mag_viz = np.log10(np.abs(fft_numpy))
fft_numpy_ang_viz = np.angle(fft_numpy)

################################################################################
# Torch

device = torch.device('cuda:0')

im_torch = torch.from_numpy(im[None,:,:])  # add batch dim
im_torch = im_torch.to(device)

# fft = complex-to-complex, rfft = real-to-complex
fft_torch = torch.rfft(im_torch, signal_ndim=2, onesided=False)
fft_torch = fftshift(fft_torch[:,:,:,0], fft_torch[:,:,:,1])
fft_torch = fft_torch.cpu().numpy().squeeze()
print(fft_torch.shape)
fft_torch = np.split(fft_torch, 2, -1)  # complex => real/imag
fft_torch = np.squeeze(fft_torch, -1)
fft_torch = fft_torch[0] + 1j*fft_torch[1]

print('fft_torch', fft_torch.shape, fft_torch.dtype)
fft_torch_mag_viz = np.log10(np.abs(fft_torch))
fft_torch_ang_viz = np.angle(fft_torch)

tolerance = 1e-6

print(fft_numpy.dtype, fft_torch.dtype)
all_close_real = np.allclose(np.real(fft_numpy), np.real(fft_torch), atol=tolerance)
all_close_imag = np.allclose(np.imag(fft_numpy), np.imag(fft_torch), atol=tolerance)

print('fftshift allclose real: {}'.format(all_close_real))
print('fftshift allclose imag: {}'.format(all_close_imag))

################################################################################

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10,6))

# Plotting NumPy results
ax[0][0].imshow(im, cmap='gray')

ax[0][1].imshow(fft_numpy_mag_viz, cmap='gray')
ax[0][1].set_title('NumPy Magnitude Spectrum')
ax[0][2].imshow(fft_numpy_ang_viz, cmap='gray')
ax[0][2].set_title('NumPy Phase Spectrum')

# Plotting PyTorch results
ax[1][0].imshow(im, cmap='gray')
ax[1][1].imshow(fft_torch_mag_viz, cmap='gray')
ax[1][1].set_title('Torch Magnitude Spectrum')
ax[1][2].imshow(fft_torch_ang_viz, cmap='gray')
ax[1][2].set_title('Torch Phase Spectrum')

for cur_ax in ax.flatten():
    cur_ax.axis('off')
plt.tight_layout()
plt.show()
