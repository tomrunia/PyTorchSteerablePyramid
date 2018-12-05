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

################################################################################

def visualize(coeff, normalize=True):
    # For visualization of all pyramid levels and orientations
    M, N = coeff[1][0].shape
    Norients = len(coeff[1])
    out = np.zeros((M * 2 - coeff[-1].shape[0], Norients * N))

    currentx = 0
    currenty = 0
    for i in range(1, len(coeff[:-1])):
        for j in range(len(coeff[1])):
            tmp = coeff[i][j].real
            m, n = tmp.shape

            if normalize:
                tmp = 255 * tmp/tmp.max()

            tmp[m - 1, :] = 255
            tmp[:, n - 1] = 2555

            out[currentx: currentx + m, currenty: currenty + n] = tmp
            currenty += n
        currentx += coeff[i][0].shape[0]
        currenty = 0

    m, n = coeff[-1].shape
    out[currentx: currentx+m, currenty: currenty+n] = 255 * coeff[-1]/coeff[-1].max()

    out[0, :] = 255
    out[:, 0] = 255

    return out

################################################################################
################################################################################

def prepare_grid(m, n):
    x = np.linspace(-(m // 2)/(m / 2), (m // 2)/(m / 2) - (1 - m % 2)*2/m, num=m)
    y = np.linspace(-(n // 2)/(n / 2), (n // 2)/(n / 2) - (1 - n % 2)*2/n, num=n)
    xv, yv = np.meshgrid(y, x)
    angle = np.arctan2(yv, xv)
    rad = np.sqrt(xv**2 + yv**2)
    rad[m//2][n//2] = rad[m//2][n//2 - 1]
    log_rad = np.log2(rad)
    return log_rad, angle

def rcosFn(width, position):
    '''
    Return a lookup table (suitable for use by INTERP1)
    containing a "raised cosine" soft threshold function
    '''
    N = 256  # abritrary
    X = np.pi * np.array(range(-N-1, 2))/2/N
    Y = np.cos(X)**2
    Y[0] = Y[1]
    Y[N+2] = Y[N+1]
    X = position + 2*width/np.pi*(X + np.pi/4)
    return X, Y

def pointOp(im, Y, X):
    ''' https://github.com/LabForComputationalVision/pyPyrTools/blob/master/pyPyrTools/pointOp.py '''
    out = np.interp(im.flatten(), X, Y)
    return np.reshape(out, im.shape)

def getlist(coeff):
    straight = [bands for scale in coeff[1:-1] for bands in scale]
    straight = [coeff[0]] + straight + [coeff[-1]]
    return straight
