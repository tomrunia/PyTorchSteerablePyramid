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
import torch

################################################################################

x = np.linspace(-10, 10, 100)
y = np.cos(np.pi*x)

################################################################################
# NumPy

y_fft_numpy = np.fft.fft(y)

################################################################################
# Torch

device = torch.device('cuda:0')

y_torch = torch.from_numpy(y[None, :])
y_torch = y_torch.to(device)

# fft = complex-to-complex, rfft = real-to-complex
y_fft_torch = torch.rfft(y_torch, signal_ndim=1, onesided=False)
y_fft_torch = y_fft_torch.cpu().numpy().squeeze()
y_fft_torch = y_fft_torch[:, 0] + 1j*y_fft_torch[:, 1]

tolerance = 1e-6
all_close = np.allclose(y_fft_numpy, y_fft_torch, atol=tolerance)
print('numpy', y_fft_numpy.shape, y_fft_numpy.dtype)
print('torch', y_fft_torch.shape, y_fft_torch.dtype)
print('Succesful: {}'.format(all_close))
