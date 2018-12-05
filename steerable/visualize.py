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
# Date Created: 2018-12-05

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

################################################################################

def visualize(coeff, example_idx=0, normalize=True):
    # Wrapper function to visualize all cases
    if isinstance(coeff[1][0], torch.Tensor):
        # PyTorch, batch of examples
        return visualize_torch_from_batch(coeff, example_idx, normalize)
    else:
        # NumPy, single example
        return visualize_single(coeff, normalize)

################################################################################

def visualize_torch_from_batch(coeff, example_idx=0, normalize=True):
    
    # For visualization of all pyramid levels and orientations
    _, M, N, _ = coeff[1][0].shape
    Norients = len(coeff[1])

    print('M,N', M,N)
    print('orientations: {}'.format(Norients))
    print('coeff[-1].shape[1]', coeff[-1].shape[1])
    
    out = np.zeros((M * 2 - coeff[-1].shape[0], Norients * N))

    currentx, currenty = 0, 0
    
    for i in range(1, len(coeff[:-1])):
        for j in range(len(coeff[1])):
        
            # Select example and real component
            tmp = coeff[i][j][example_idx,:,:,0]
            tmp = tmp.cpu().numpy()

            m, n = tmp.shape
            print('tmp.shape', tmp.shape)

            if normalize:
                tmp = 255 * tmp/tmp.max()

            tmp[m-1,:] = 255
            tmp[:,n-1] = 255

            out[currentx:currentx+m, currenty:currenty+n] = tmp
            currenty += n

        currentx += coeff[i][0].shape[1]
        currenty = 0

    # Low-pass
    _, m, n, _ = coeff[-1].shape
    tmp = coeff[-1][example_idx,:,:,0]  # select batch and real part
    tmp = tmp.cpu().numpy()
    out[currentx: currentx+m, currenty: currenty+n] = 255 * tmp/tmp.max()

    out[0, :] = 255
    out[:, 0] = 255
    return out

################################################################################

def visualize_single(coeff, normalize=True):

    # For visualization of all pyramid levels and orientations
    M, N = coeff[1][0].shape
    Norients = len(coeff[1])
    out = np.zeros((M * 2 - coeff[-1].shape[0], Norients * N))

    currentx, currenty = 0, 0

    print('i', 1, len(coeff[:-1]))
    print('j', 0, len(coeff[1]))

    for i in range(1, len(coeff[:-1])):
        for j in range(len(coeff[1])):
            tmp = coeff[i][j].real
            m, n = tmp.shape
            print('tmp.shape', tmp.shape)

            if normalize:
                tmp = 255 * tmp/tmp.max()

            tmp[m-1,:] = 255
            tmp[:,n-1] = 255
            out[currentx:currentx+m,currenty:currenty+n] = tmp
            currenty += n
        
        print('offset', coeff[i][0].shape[0])
        currentx += coeff[i][0].shape[0]
        currenty = 0

    m, n = coeff[-1].shape
    out[currentx: currentx+m, currenty: currenty+n] = 255 * coeff[-1]/coeff[-1].max()

    out[0,:] = 255
    out[:,0] = 255

    return out