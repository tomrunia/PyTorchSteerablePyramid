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

from steerable.SCFpyr_PyTorch import SCFpyr_PyTorch
from steerable.visualize import visualize

import cortex.vision
import cv2

################################################################################
################################################################################
    
device = torch.device('cuda:0')
batch_size = 1

image_file = './assets/lena.jpg'
im = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
im = cortex.vision.resize(im, out_height=200, out_width=200)
im = im.astype(np.float64)/255.
im_batch = np.tile(im, (batch_size,1,1))

# Move to Torch on the GPU
im_batch = torch.from_numpy(im_batch)
im_batch = im_batch.to(device)

# Build the complex steerable pyramid
pyr = SCFpyr_PyTorch(height=5, scale_factor=2, device=device)
coeff_batch = pyr.build(im_batch)

# HighPass:         coeff[0] : highpass
# BandPass Scale 1: coeff[1][0], coeff[1][1], coeff[1][2], coeff[1][3]
# BandPass Scale 2: coeff[2][0], coeff[2][1], coeff[2][2], coeff[2][3]

filter_viz = visualize(coeff_batch, example_idx=0)
cv2.imshow('filter visualization', filter_viz)
cv2.waitKey(0)

