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

import argparse
import numpy as np
import skimage
import time
import cv2

from steerable.SCFpyr_NumPy import SCFpyr_NumPy
import steerable.utils as utils

################################################################################
################################################################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_file', type=str, default='./assets/patagonia.jpg')
    parser.add_argument('--batch_size', type=int, default='1')
    parser.add_argument('--image_size', type=int, default='200')
    parser.add_argument('--pyr_nlevels', type=int, default='5')
    parser.add_argument('--pyr_nbands', type=int, default='4')
    parser.add_argument('--pyr_scale_factor', type=int, default='2')
    parser.add_argument('--visualize', type=bool, default=True)
    config = parser.parse_args()

    ############################################################################
    # Build the complex steerable pyramid

    pyr = SCFpyr_NumPy(
        height=config.pyr_nlevels, 
        nbands=config.pyr_nbands,
        scale_factor=config.pyr_scale_factor, 
    )

    ############################################################################
    # Create a batch and feed-forward

    start_time = time.time()

    # Load Batch
    #im_batch_numpy = skimage.io.imread('./assets/lena.jpg', as_gray=True)
    #im_batch_numpy = im_batch_numpy[None,:,:]  # expand with batch dim

    im_batch_numpy = utils.load_image_batch(config.image_file, config.batch_size, config.image_size)
    im_batch_numpy = im_batch_numpy.squeeze(1)  # no channel dim for NumPy

    # Compute Steerable Pyramid
    start_time = time.time()
    for image in im_batch_numpy:
        cv2.imshow('image', image)
        coeff = pyr.build(image)

    
    print(0, coeff.level_size(0))
    print(1, coeff.level_size(1))
    print(2, coeff.level_size(2))
    print(3, coeff.level_size(3))

    exit()

    
    duration = time.time()-start_time
    print('Finishing decomposing {batch_size} images in {duration:.1f} seconds.'.format(
        batch_size=config.batch_size,
        duration=duration
    ))

    ############################################################################
    # Visualization

    print('#'*60)
    for i in range(len(coeff)):
        if isinstance(coeff[i], list):
            print('level', i, len(coeff[i]), type(coeff[i]), coeff[i][0].shape, coeff[i][0].dtype)
        else:
            label = 'highpass' if i == 0 else 'lowpass'
            print(label, len(coeff[i]), type(coeff[i]), coeff[i].shape, coeff[i].dtype)
    print('#'*60)

    if config.visualize:
        import cv2
        filter_viz = utils.make_grid_coeff(coeff, normalize=True)
        cv2.imshow('coeff', filter_viz)
        cv2.waitKey(0)
        
