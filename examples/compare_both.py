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

from steerable.SCFpyr import SCFpyr
from steerable.SCFpyr_PyTorch import SCFpyr_PyTorch
from steerable.visualize import visualize

import cortex.vision
import cv2

################################################################################
################################################################################
# Common

image_file = '/home/tomrunia/data/lena.jpg'
im = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
im = cortex.vision.resize(im, out_height=200, out_width=200)
im = im.astype(np.float32)/255.

# Number of pyramid levels
pyr_height = 5

################################################################################
################################################################################
# NumPy

# Build the complex steerable pyramid
pyr = SCFpyr(height=pyr_height)
coeff_numpy = pyr.build(im)

################################################################################
# PyTorch

device = torch.device('cuda:0')
batch_size = 16

# Create a batch of images
im_batch = np.tile(im, (batch_size,1,1))

# Move to Torch on the GPU
im_batch = torch.from_numpy(im_batch)
im_batch = im_batch.to(device)

# Build the complex steerable pyramid
pyr = SCFpyr_PyTorch(height=pyr_height, scale_factor=2, device=device)
coeff_torch = pyr.build(im_batch)

# HighPass:         coeff[0] : highpass
# BandPass Scale 1: coeff[1][0], coeff[1][1], coeff[1][2], coeff[1][3]
# BandPass Scale 2: coeff[2][0], coeff[2][1], coeff[2][2], coeff[2][3]

################################################################################

bands_viz_numpy = visualize(coeff_numpy)
bands_viz_torch = visualize(coeff_torch, example_idx=0)

cv2.imshow('NumPy Bands', bands_viz_numpy)
cv2.imshow('PyTorch Bands', bands_viz_torch)
cv2.waitKey(0)



