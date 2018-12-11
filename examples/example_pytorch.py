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
import time
import torch

from steerable.SCFpyr_PyTorch import SCFpyr_PyTorch
import steerable.utils as utils

################################################################################
################################################################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_file', type=str, default='./assets/patagonia.jpg')
    parser.add_argument('--batch_size', type=int, default='32')
    parser.add_argument('--image_size', type=int, default='200')
    parser.add_argument('--pyr_nlevels', type=int, default='5')
    parser.add_argument('--pyr_nbands', type=int, default='4')
    parser.add_argument('--pyr_scale_factor', type=int, default='2')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--visualize', type=bool, default=True)
    config = parser.parse_args()

    device = utils.get_device(config.device)

    ############################################################################
    # Build the complex steerable pyramid

    pyr = SCFpyr_PyTorch(
        height=config.pyr_nlevels, 
        nbands=config.pyr_nbands,
        scale_factor=config.pyr_scale_factor, 
        device=device
    )

    ############################################################################
    # Create a batch and feed-forward

    start_time = time.time()

    # Load Batch
    im_batch_numpy = utils.load_image_batch(config.image_file, config.batch_size, config.image_size)
    im_batch_torch = torch.from_numpy(im_batch_numpy).to(device)

    # Compute Steerable Pyramid
    coeff = pyr.build(im_batch_torch)

    duration = time.time()-start_time
    print('Finishing decomposing {batch_size} images in {duration:.1f} seconds.'.format(
        batch_size=config.batch_size,
        duration=duration
    ))

    ############################################################################
    # Visualization

    # Just extract a single example from the batch
    # Also moves the example to CPU and NumPy
    coeff = utils.extract_from_batch(coeff, 0)

    if config.visualize:
        import cv2
        coeff_grid = utils.make_grid_coeff(coeff, normalize=True)
        cv2.imshow('image', im_batch_numpy[0,0,])
        cv2.imshow('coeff', coeff_grid)
        cv2.waitKey(0)
        