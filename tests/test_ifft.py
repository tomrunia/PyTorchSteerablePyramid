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
# Date Created: 2018-12-07

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import cv2

import steerable.fft as fft_utils
import matplotlib.pyplot as plt

################################################################################

tolerance = 1e-6

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

ifft_numpy1 = np.fft.ifftshift(fft_numpy) 
ifft_numpy = np.fft.ifft2(ifft_numpy1)

################################################################################
# Torch

device = torch.device('cpu')

im_torch = torch.from_numpy(im[None,:,:])  # add batch dim
im_torch = im_torch.to(device)

# fft = complex-to-complex, rfft = real-to-complex
fft_torch = torch.rfft(im_torch, signal_ndim=2, onesided=False)
fft_torch = fft_utils.batch_fftshift2d(fft_torch)

ifft_torch = fft_utils.batch_ifftshift2d(fft_torch)
ifft_torch = torch.ifft(ifft_torch, signal_ndim=2, normalized=False)

ifft_torch_to_numpy = ifft_torch.numpy()
ifft_torch_to_numpy = np.split(ifft_torch_to_numpy, 2, -1)  # complex => real/imag
ifft_torch_to_numpy = np.squeeze(ifft_torch_to_numpy, -1)
ifft_torch_to_numpy = ifft_torch_to_numpy[0] + 1j*ifft_torch_to_numpy[1]
all_close_ifft = np.allclose(ifft_numpy1, ifft_torch_to_numpy, atol=tolerance)
print('ifft all close: ', all_close_ifft)

fft_torch = fft_torch.cpu().numpy().squeeze()
fft_torch = np.split(fft_torch, 2, -1)  # complex => real/imag
fft_torch = np.squeeze(fft_torch, -1)
fft_torch = fft_torch[0] + 1j*fft_torch[1]

ifft_torch = ifft_torch.cpu().numpy().squeeze()
ifft_torch = np.split(ifft_torch, 2, -1)  # complex => real/imag
ifft_torch = np.squeeze(ifft_torch, -1)
ifft_torch = ifft_torch[0] + 1j*ifft_torch[1]

fft_torch_mag_viz = np.log10(np.abs(fft_torch))
fft_torch_ang_viz = np.angle(fft_torch)

################################################################################
# Tolerance checking

all_close_real = np.allclose(np.real(fft_numpy), np.real(fft_torch), atol=tolerance)
all_close_imag = np.allclose(np.imag(fft_numpy), np.imag(fft_torch), atol=tolerance)
print('fft allclose real: {}'.format(all_close_real))
print('fft allclose imag: {}'.format(all_close_imag))

all_close_real = np.allclose(np.real(ifft_numpy), np.real(ifft_torch), atol=tolerance)
all_close_imag = np.allclose(np.imag(ifft_numpy), np.imag(ifft_torch), atol=tolerance)
print('ifft allclose real: {}'.format(all_close_real))
print('ifft allclose imag: {}'.format(all_close_imag))

################################################################################

fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(12,6))

# Plotting NumPy results
ax[0][0].imshow(im, cmap='gray')

ax[0][1].imshow(fft_numpy_mag_viz, cmap='gray')
ax[0][1].set_title('NumPy fft magnitude')
ax[0][2].imshow(fft_numpy_ang_viz, cmap='gray')
ax[0][2].set_title('NumPy fft spectrum')
ax[0][3].imshow(ifft_numpy.real, cmap='gray')
ax[0][3].set_title('NumPy ifft real')
ax[0][4].imshow(ifft_numpy.imag, cmap='gray')
ax[0][4].set_title('NumPy ifft imag')

# Plotting PyTorch results
ax[1][0].imshow(im, cmap='gray')
ax[1][1].imshow(fft_torch_mag_viz, cmap='gray')
ax[1][1].set_title('PyTorch fft magnitude')
ax[1][2].imshow(fft_torch_ang_viz, cmap='gray')
ax[1][2].set_title('PyTorch fft phase')
ax[1][3].imshow(ifft_torch.real, cmap='gray')
ax[1][3].set_title('PyTorch ifft real')
ax[1][4].imshow(ifft_torch.imag, cmap='gray')
ax[1][4].set_title('PyTorch ifft imag')

for cur_ax in ax.flatten():
    cur_ax.axis('off')
plt.tight_layout()
plt.show()
