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
# Date Created: 2018-12-12

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import cortex.vision

numpy_outdft = np.load('./assets/numpy_outdft_real.npy') + 1j*np.load('./assets/numpy_outdft_imag.npy')
torch_outdft = np.load('./assets/torch_outdft_real.npy') + 1j*np.load('./assets/torch_outdft_imag.npy')

print('allclose, real', np.allclose(numpy_outdft.real, torch_outdft.real, atol=1e-2))
print('allclose, imag', np.allclose(numpy_outdft.imag, torch_outdft.imag, atol=1e-2))

real, imag = numpy_outdft.real, numpy_outdft.imag
print('  [numpy] outdft real ({:.3f}, {:.3f}, {:.3f})'.format(
    real[0:20,0:20].min().item(), real[0:20,0:20].mean().item(), real[0:20,0:20].max().item()  
))
print('  [numpy] outdft imag ({:.3f}, {:.3f}, {:.3f})'.format(
    imag[0:20,0:20].min().item(), imag[0:20,0:20].mean().item(), imag[0:20,0:20].max().item()  
))

real, imag = torch_outdft.real, torch_outdft.imag
print('  [torch] outdft real ({:.3f}, {:.3f}, {:.3f})'.format(
    real[0:20,0:20].min().item(), real[0:20,0:20].mean().item(), real[0:20,0:20].max().item()  
))
print('  [torch] outdft imag ({:.3f}, {:.3f}, {:.3f})'.format(
    imag[0:20,0:20].min().item(), imag[0:20,0:20].mean().item(), imag[0:20,0:20].max().item()  
))

numpy_reconstruction = np.fft.ifftshift(numpy_outdft)
numpy_reconstruction = np.fft.ifft2(numpy_reconstruction)
numpy_reconstruction = numpy_reconstruction.real
real = numpy_reconstruction
print('  [numpy] outdft real ({:.3f}, {:.3f}, {:.3f})'.format(
    real[0:20,0:20].min().item(), real[0:20,0:20].mean().item(), real[0:20,0:20].max().item()  
))

torch_reconstruction = np.fft.ifftshift(torch_outdft)
torch_reconstruction = np.fft.ifft2(torch_reconstruction)
torch_reconstruction = torch_reconstruction.real
real = torch_reconstruction
print('  [torch] outdft real ({:.3f}, {:.3f}, {:.3f})'.format(
    real[0:20,0:20].min().item(), real[0:20,0:20].mean().item(), real[0:20,0:20].max().item()  
))

cv2.imshow('numpy phase', np.angle(numpy_outdft))
cv2.imshow('numpy imag', np.abs(numpy_outdft))
cv2.imshow('torch phase', np.angle(torch_outdft))
cv2.imshow('torch imag', np.abs(torch_outdft))

cv2.imshow('numpy reconstruction', numpy_reconstruction.astype(np.uint8))
cv2.imshow('torch reconstruction', torch_reconstruction.astype(np.uint8))

cv2.waitKey(0)