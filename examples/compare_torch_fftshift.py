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

from steerable.fft import fftshift

################################################################################

x = np.linspace(-10, 10, 100)
y = np.cos(np.pi*x)

################################################################################
# NumPy

y_fft_numpy = np.fft.fft(y)
y_fft_numpy_shift = np.fft.fftshift(y_fft_numpy)
fft_numpy_real = np.real(y_fft_numpy_shift)
fft_numpy_imag = np.imag(y_fft_numpy_shift)

################################################################################
# Torch

device = torch.device('cuda:0')

y_torch = torch.from_numpy(y[None, :])
y_torch = y_torch.to(device)

# fft = complex-to-complex, rfft = real-to-complex
y_fft_torch = torch.rfft(y_torch, signal_ndim=1, onesided=False)
y_fft_torch = fftshift(y_fft_torch[:,:,0], y_fft_torch[:,:,1])
y_fft_torch = y_fft_torch.cpu().numpy().squeeze()
fft_torch_real = y_fft_torch[:,0]
fft_torch_imag = y_fft_torch[:,1]

tolerance = 1e-6

all_close_real = np.allclose(fft_numpy_real, fft_torch_real, atol=tolerance)
all_close_imag = np.allclose(fft_numpy_imag, fft_torch_imag, atol=tolerance)

print('fftshift allclose real: {}'.format(all_close_real))
print('fftshift allclose imag: {}'.format(all_close_imag))

################################################################################

import cortex.plot
import cortex.plot.colors
colors = cortex.plot.nature_colors()

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10,6))

# Plotting NumPy results
ax[0][0].plot(x, y, color=colors[3])
ax[0][1].plot(np.real(y_fft_numpy), color=colors[0])
ax[0][1].plot(fft_numpy_real, color=colors[1])
ax[0][1].set_title('NumPy Real')
ax[0][2].plot(np.imag(y_fft_numpy), color=colors[0])
ax[0][2].plot(fft_numpy_real, color=colors[1])
ax[0][2].set_title('NumPy Imag')

# Plotting NumPy results
ax[1][0].plot(x, y, color=colors[3])
ax[1][1].plot(fft_torch_real, color=colors[1])
ax[1][1].set_title('Torch Real')
ax[1][2].plot(fft_torch_imag, color=colors[1])
ax[1][2].set_title('Torch Imag')

plt.show()
