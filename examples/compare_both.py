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
import cv2

from steerable.SCFpyr_NumPy import SCFpyr_NumPy
from steerable.SCFpyr_PyTorch import SCFpyr_PyTorch
import steerable.utils as utils

################################################################################
################################################################################
# Common

image_file = './assets/lena.jpg'
image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (200,200))

# Number of pyramid levels
pyr_height = 5

# Number of orientation bands
pyr_nbands = 4

# Tolerance for error checking
tolerance = 1e-3

################################################################################
# NumPy

pyr_numpy = SCFpyr_NumPy(pyr_height, pyr_nbands, scale_factor=2)
coeff_numpy = pyr_numpy.build(image)
reconstruction_numpy = pyr_numpy.reconstruct(coeff_numpy)

print('#'*60)

################################################################################
# PyTorch

device = torch.device('cuda:0')

im_batch = torch.from_numpy(image[None,None,:,:])
im_batch = im_batch.to(device).float()

pyr_torch = SCFpyr_PyTorch(pyr_height, pyr_nbands, device=device)
coeff_torch = pyr_torch.build(im_batch)
reconstruction_torch = pyr_torch.reconstruct(coeff_torch)
reconstruction_torch = reconstruction_torch.cpu().numpy()

# Just extract a single example from the batch
# Also moves the example to CPU and NumPy
coeff_torch = utils.extract_from_batch(coeff_torch, 0)

cv2.waitKey(0)
exit()

################################################################################
# Check correctness

print('#'*60)
assert len(coeff_numpy) == len(coeff_torch)

for level, _ in enumerate(coeff_numpy):

    print('Pyramid Level {level}'.format(level=level))
    coeff_level_numpy = coeff_numpy[level]
    coeff_level_torch = coeff_torch[level]

    assert isinstance(coeff_level_torch, type(coeff_level_numpy))
    
    if isinstance(coeff_level_numpy, np.ndarray):

        # Low- or High-Pass
        print('  NumPy.   min = {min:.3f}, max = {max:.3f},'
              ' mean = {mean:.3f}, std = {std:.3f}'.format(
                  min=np.min(coeff_level_numpy), max=np.max(coeff_level_numpy), 
                  mean=np.mean(coeff_level_numpy), std=np.std(coeff_level_numpy)))

        print('  PyTorch. min = {min:.3f}, max = {max:.3f},'
              ' mean = {mean:.3f}, std = {std:.3f}'.format(
                  min=np.min(coeff_level_torch), max=np.max(coeff_level_torch), 
                  mean=np.mean(coeff_level_torch), std=np.std(coeff_level_torch)))

        # Check numerical correctness
        assert np.allclose(coeff_level_numpy, coeff_level_torch, atol=tolerance)

    elif isinstance(coeff_level_numpy, list):

        # Intermediate bands
        for band, _ in enumerate(coeff_level_numpy):

            band_numpy = coeff_level_numpy[band]
            band_torch = coeff_level_torch[band]

            print('  Orientation Band {}'.format(band))
            print('    NumPy.   min = {min:.3f}, max = {max:.3f},'
                  ' mean = {mean:.3f}, std = {std:.3f}'.format(
                      min=np.min(band_numpy), max=np.max(band_numpy), 
                      mean=np.mean(band_numpy), std=np.std(band_numpy)))

            print('    PyTorch. min = {min:.3f}, max = {max:.3f},'
                  ' mean = {mean:.3f}, std = {std:.3f}'.format(
                      min=np.min(band_torch), max=np.max(band_torch), 
                      mean=np.mean(band_torch), std=np.std(band_torch)))

            # Check numerical correctness
            assert np.allclose(band_numpy, band_torch, atol=tolerance)

################################################################################
# Visualize

coeff_grid_numpy = utils.make_grid_coeff(coeff_numpy, normalize=True)
coeff_grid_torch = utils.make_grid_coeff(coeff_torch, normalize=True)

import cortex.vision

reconstruction_torch = np.ascontiguousarray(reconstruction_torch[0], np.float32)
reconstruction_numpy = np.ascontiguousarray(reconstruction_numpy, np.float32)

reconstruction_torch = cortex.vision.normalize_for_display(reconstruction_torch)
reconstruction_numpy = cortex.vision.normalize_for_display(reconstruction_numpy)

cv2.imshow('image', image)
cv2.imshow('coeff numpy', coeff_grid_numpy)
cv2.imshow('coeff torch', coeff_grid_torch)
cv2.imshow('reconstruction numpy', reconstruction_numpy)
cv2.imshow('reconstruction torch', reconstruction_torch)

cv2.waitKey(0)
