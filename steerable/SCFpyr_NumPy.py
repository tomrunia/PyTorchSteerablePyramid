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
from scipy.misc import factorial

import steerable.math_utils as math_utils
pointOp = math_utils.pointOp
        
################################################################################

class SCFpyr_NumPy():
    '''
    This is a modified version of buildSFpyr, that constructs a
    complex-valued steerable pyramid  using Hilbert-transform pairs
    of filters. Note that the imaginary parts will *not* be steerable.

    Description of this transform appears in: Portilla & Simoncelli,
    International Journal of Computer Vision, 40(1):49-71, Oct 2000.
    Further information: http://www.cns.nyu.edu/~eero/STEERPYR/

    Modified code from the perceptual repository:
      https://github.com/andreydung/Steerable-filter

    This code looks very similar to the original Matlab code:
      https://github.com/LabForComputationalVision/matlabPyrTools/blob/master/buildSCFpyr.m

    Also looks very similar to the original Python code presented here:
      https://github.com/LabForComputationalVision/pyPyrTools/blob/master/pyPyrTools/SCFpyr.py

    '''

    def __init__(self, height=5, nbands=4, scale_factor=2):
        self.nbands  = nbands  # number of orientation bands
        self.height  = height  # including low-pass and high-pass
        self.scale_factor = scale_factor
        
        # Cache constants
        self.lutsize = 1024
        self.Xcosn = np.pi * np.array(range(-(2*self.lutsize+1), (self.lutsize+2)))/self.lutsize
        self.alpha = (self.Xcosn + np.pi) % (2*np.pi) - np.pi


    ################################################################################
    # Construction of Steerable Pyramid

    def build(self, im):
        ''' Decomposes an image into it's complex steerable pyramid. 
        The pyramid typically has ~4 levels and 4-8 orientations. 
        
        Args:
            im_batch (np.ndarray): single image [H,W]
        
        Returns:
            pyramid: list containing np.ndarray objects storing the pyramid
        '''

        assert len(im.shape) == 2, 'Input im must be grayscale'
        height, width = im.shape

        # Check whether image size is sufficient for number of levels
        if self.height > int(np.floor(np.log2(min(width, height))) - 2):
            raise RuntimeError('Cannot build {} levels, image too small.'.format(self.height))
        
        # Prepare a grid
        log_rad, angle = math_utils.prepare_grid(height, width)

        # Radial transition function (a raised cosine in log-frequency):
        Xrcos, Yrcos = math_utils.rcosFn(1, -0.5)
        Yrcos = np.sqrt(Yrcos)

        YIrcos = np.sqrt(1 - Yrcos**2)
        lo0mask = pointOp(log_rad, YIrcos, Xrcos)
        hi0mask = pointOp(log_rad, Yrcos, Xrcos)

        # Shift the zero-frequency component to the center of the spectrum.
        imdft = np.fft.fftshift(np.fft.fft2(im))

        # Low-pass
        lo0dft = imdft * lo0mask

        # Recursive build the steerable pyramid
        coeff = self._build_levels(lo0dft, log_rad, angle, Xrcos, Yrcos, self.height-1)

        # High-pass
        hi0dft = imdft * hi0mask
        hi0 = np.fft.ifft2(np.fft.ifftshift(hi0dft))
        coeff.insert(0, hi0.real)
        return coeff


    def _build_levels(self, lodft, log_rad, angle, Xrcos, Yrcos, height):

        if height <= 1:

            # Low-pass
            lo0 = np.fft.ifftshift(lodft)
            lo0 = np.fft.ifft2(lo0)
            coeff = [lo0.real]

        else:
            
            Xrcos = Xrcos - np.log2(self.scale_factor)

            ####################################################################
            ####################### Orientation bandpass #######################
            ####################################################################

            himask = pointOp(log_rad, Yrcos, Xrcos)

            order = self.nbands - 1
            const = np.power(2, 2*order) * np.square(factorial(order)) / (self.nbands * factorial(2*order))
            Ycosn = 2*np.sqrt(const) * np.power(np.cos(self.Xcosn), order) * (np.abs(self.alpha) < np.pi/2)

            # Loop through all orientation bands
            orientations = []
            for b in range(self.nbands):
                anglemask = pointOp(angle, Ycosn, self.Xcosn + np.pi*b/self.nbands)
                banddft = np.power(np.complex(0, -1), self.nbands - 1) * lodft * anglemask * himask
                band = np.fft.ifft2(np.fft.ifftshift(banddft))
                orientations.append(band)

            ####################################################################
            ######################## Subsample lowpass #########################
            ####################################################################

            dims = np.array(lodft.shape)

            # Both are tuples of size 2
            low_ind_start = (np.ceil((dims+0.5)/2) - np.ceil((np.ceil((dims-0.5)/2)+0.5)/2)).astype(int)
            low_ind_end   = (low_ind_start + np.ceil((dims-0.5)/2)).astype(int)
          
            # Selection
            log_rad = log_rad[low_ind_start[0]:low_ind_end[0], low_ind_start[1]:low_ind_end[1]]
            angle   = angle[low_ind_start[0]:low_ind_end[0], low_ind_start[1]:low_ind_end[1]]
            lodft   = lodft[low_ind_start[0]:low_ind_end[0], low_ind_start[1]:low_ind_end[1]]

            # Subsampling in frequency domain
            YIrcos = np.abs(np.sqrt(1 - Yrcos**2))
            lomask = pointOp(log_rad, YIrcos, Xrcos)
            lodft = lomask * lodft

            ####################################################################
            ####################### Recursion next level #######################
            ####################################################################

            coeff = self._build_levels(lodft, log_rad, angle, Xrcos, Yrcos, height-1)
            coeff.insert(0, orientations)

        return coeff

    ################################################################################
    # Reconstruction to Image

    def reconstruct(self, coeff):

        if self.nbands != len(coeff[1]):
            raise Exception("Unmatched number of orientations")

        M, N = coeff[0].shape
        log_rad, angle = math_utils.prepare_grid(M, N)

        Xrcos, Yrcos = math_utils.rcosFn(1, -0.5)
        Yrcos  = np.sqrt(Yrcos)
        YIrcos = np.sqrt(np.abs(1 - Yrcos*Yrcos))

        lo0mask = pointOp(log_rad, YIrcos, Xrcos)
        hi0mask = pointOp(log_rad, Yrcos, Xrcos)

        tempdft = self._reconstruct_levels(coeff[1:], log_rad, Xrcos, Yrcos, angle)

        hidft = np.fft.fftshift(np.fft.fft2(coeff[0]))
        outdft = tempdft * lo0mask + hidft * hi0mask

        reconstruction = np.fft.ifft2(np.fft.ifftshift(outdft))
        reconstruction = reconstruction.real.astype(int) 
        return reconstruction

    def _reconstruct_levels(self, coeff, log_rad, Xrcos, Yrcos, angle):
        
        print('[numpy] Call to _reconstruct_levels. remaining = {rem}'.format(rem=len(coeff)))

        if len(coeff) == 1:
            print('[numpy] len(coeff)==1')
            print('[numpy] coeff[0].shape', coeff[0].shape, coeff[0].dtype)
            dft = np.fft.fft2(coeff[0])
            print('[torch] dft after fft', dft.shape, dft.dtype)
            tmp = np.fft.fftshift(dft)
            print('[torch] dft after fftshift', dft.shape, dft.dtype)
            print('[numpy] here!!')
            return tmp

        Xrcos = Xrcos - np.log2(self.scale_factor)

        ####################################################################
        ####################### Orientation Residue ########################
        ####################################################################

        himask = pointOp(log_rad, Yrcos, Xrcos)

        lutsize = 1024
        Xcosn = np.pi * np.array(range(-(2*lutsize+1), (lutsize+2)))/lutsize
        order = self.nbands - 1
        const = np.power(2, 2*order) * np.square(factorial(order)) / (self.nbands * factorial(2*order))
        Ycosn = np.sqrt(const) * np.power(np.cos(Xcosn), order)

        orientdft = np.zeros(coeff[0][0].shape)

        for b in range(self.nbands):

            anglemask = pointOp(angle, Ycosn, Xcosn + np.pi * b/self.nbands)

            banddft = np.fft.fftshift(np.fft.fft2(coeff[0][b]))
            orientdft = orientdft + np.power(np.complex(0, 1), order) * banddft * anglemask * himask

        ####################################################################
        ########## Lowpass component are upsampled and convoluted ##########
        ####################################################################

        dims = np.array(coeff[0][0].shape)

        lostart = (np.ceil((dims+0.5)/2) -
                    np.ceil((np.ceil((dims-0.5)/2)+0.5)/2)).astype(np.int32)
        loend = lostart + np.ceil((dims-0.5)/2).astype(np.int32)

        nlog_rad = log_rad[lostart[0]:loend[0], lostart[1]:loend[1]]
        nangle = angle[lostart[0]:loend[0], lostart[1]:loend[1]]
        YIrcos = np.sqrt(np.abs(1 - Yrcos * Yrcos))
        lomask = pointOp(nlog_rad, YIrcos, Xrcos)

        nresdft = self._reconstruct_levels(coeff[1:], nlog_rad, Xrcos, Yrcos, nangle)

        print('  [numpy] nresdft', nresdft.shape, nresdft.dtype)
        print('  [numpy] lomask', lomask.shape, lomask.dtype)

        resdft = np.zeros(dims, 'complex')
        print('  [numpy] resdft', resdft.shape, resdft.dtype)
        print('  [numpy] lostart[0] - loend[0]', lostart[0], loend[0])
        print('  [numpy] lostart[1] - loend[1]', lostart[1], loend[1])

        resdft[lostart[0]:loend[0], lostart[1]:loend[1]] = nresdft * lomask



        return resdft + orientdft


################################################################################
################################################################################
# Work in Progress

class ComplexSteerablePyramid():

    def __init__(self, height, nbands):
        self._height = height  # including low-pass and high-pass
        self._nbands = nbands  # number of orientation bands
        self._coeff = [None]*self._height
    
    def set_level(self, level, coeff):
        self._coeff[level] = coeff
    
    def get_level(self, level):
        return self._coeff[level]

    def level_size(self, level):
        if level == 0:
            # High-pass
            return self._coeff[level].shape
        elif level == self._nbands:
            # Low-pass
            return self._coeff[level][0].shape
        # Intermediate levels
        return self._coeff[level][0].shape
