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

from scipy.misc import factorial

import steerable.math_utils as math_utils
pointOp = math_utils.pointOp

################################################################################


class SCFpyr_PyTorch(object):
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

    def __init__(self, height=5, nbands=4, scale_factor=2, device=None):
        self.height = height  # including low-pass and high-pass
        self.nbands = nbands  # number of orientation bands
        self.scale_factor = scale_factor
        self.device = torch.device('cpu') if device is None else device

        # Cache constants
        self.lutsize = 1024
        self.Xcosn = np.pi * np.array(range(-(2*self.lutsize+1), (self.lutsize+2)))/self.lutsize
        self.alpha = (self.Xcosn + np.pi) % (2*np.pi) - np.pi
        self.complex_factor = np.power(np.complex(0, -1), self.nbands - 1)
        
    ################################################################################
    # Construction of Steerable Pyramid

    def build(self, im_batch):
        ''' Decomposes a batch of images into a complex steerable pyramid. 
        The pyramid typically has ~4 levels and 4-8 orientations. 
        
        Args:
            im_batch (torch.Tensor): Batch of images of shape [N,C,H,W]
        
        Returns:
            pyramid: list containing torch.Tensor objects storing the pyramid
        '''
        assert im_batch.dtype == torch.float32, 'Image batch must be torch.float32'
        assert im_batch.dim() == 4, 'Image batch must be of shape [N,C,H,W]'
        assert im_batch.shape[1] == 1, 'Second dimension must be 1 encoding grayscale image'

        height, width = im_batch.shape[2], im_batch.shape[2] 
        im_batch = im_batch.squeeze(1)  # flatten channels dim

        # Check whether im shape allows the pyramid M
        max_height_pyr = int(np.floor(np.log2(min(width, height))) - 2)
        assert max_height_pyr >= self.height, 'Cannot buid pyramid with more than {} levels'.format(max_height_pyr)
        
        # Prepare a grid
        log_rad, angle = math_utils.prepare_grid(height, width)

        # Radial transition function (a raised cosine in log-frequency):
        Xrcos, Yrcos = math_utils.rcosFn(1, -0.5)
        Yrcos = np.sqrt(Yrcos)

        YIrcos = np.sqrt(1 - Yrcos**2)

        lo0mask = pointOp(log_rad, YIrcos, Xrcos)
        hi0mask = pointOp(log_rad, Yrcos, Xrcos)

        # Note that we expand dims to support broadcasting later
        lo0mask = torch.from_numpy(lo0mask).float()[None,:,:,None].to(self.device)
        hi0mask = torch.from_numpy(hi0mask).float()[None,:,:,None].to(self.device)

        # Fourier transform (2D) and shifting
        batch_dft = torch.rfft(im_batch, signal_ndim=2, onesided=False)
        batch_dft = math_utils.batch_fftshift2d(batch_dft)

        # Low-pass
        lo0dft = batch_dft * lo0mask

        # Start recursively building the pyramids
        coeff = self._build_levels(lo0dft, log_rad, angle, Xrcos, Yrcos, self.height-1)

        # High-pass
        hi0dft = batch_dft * hi0mask

        hi0 = torch.ifft(hi0dft, signal_ndim=2)
        hi0 = math_utils.batch_ifftshift2d(hi0)
        hi0_real = torch.unbind(hi0, -1)[0]
        coeff.insert(0, hi0_real)

        return coeff


    def _build_levels(self, lodft, log_rad, angle, Xrcos, Yrcos, height):
        
        if height <= 1:

            # Low-pass
            lo0 = torch.ifft(lodft, signal_ndim=2)
            lo0 = math_utils.batch_fftshift2d(lo0)
            lo0_real = torch.unbind(lo0, -1)[0]
            # TODO: check correctess of these ops...
            coeff = [lo0_real]

        else:
            
            Xrcos = Xrcos - np.log2(self.scale_factor)

            ####################################################################
            ####################### Orientation bandpass #######################
            ####################################################################

            himask = pointOp(log_rad, Yrcos, Xrcos)
            himask = torch.from_numpy(himask[None,:,:,None]).float().to(self.device)

            order = self.nbands - 1
            const = np.power(2, 2*order) * np.square(factorial(order)) / (self.nbands * factorial(2*order))
            Ycosn = 2*np.sqrt(const) * np.power(np.cos(self.Xcosn), order) * (np.abs(self.alpha) < np.pi/2) # [n,]

            # Loop through all orientation bands
            orientations = []
            for b in range(self.nbands):

                anglemask = pointOp(angle, Ycosn, self.Xcosn + np.pi*b/self.nbands)
                anglemask = anglemask[None,:,:,None]  # for broadcasting
                anglemask = torch.from_numpy(anglemask).float().to(self.device)

                # Bandpass filtering                
                banddft = lodft * anglemask * himask

                # Now multiply with complex number
                # (x+yi)(u+vi) = (xu-yv) + (xv+yu)i
                banddft = torch.unbind(banddft, -1)
                banddft_real = self.complex_factor.real*banddft[0] - self.complex_factor.imag*banddft[1]
                banddft_imag = self.complex_factor.real*banddft[1] + self.complex_factor.imag*banddft[0]
                banddft = torch.stack((banddft_real, banddft_imag), -1)

                band = math_utils.batch_ifftshift2d(banddft)
                band = torch.ifft(band, signal_ndim=2)
                orientations.append(band)

            ####################################################################
            ######################## Subsample lowpass #########################
            ####################################################################

            # Don't consider batch_size and imag/real dim
            dims = np.array(lodft.shape[1:3])  

            # Both are tuples of size 2
            low_ind_start = (np.ceil((dims+0.5)/2) - np.ceil((np.ceil((dims-0.5)/2)+0.5)/2)).astype(int)
            low_ind_end   = (low_ind_start + np.ceil((dims-0.5)/2)).astype(int)

            # Subsampling indices
            log_rad = log_rad[low_ind_start[0]:low_ind_end[0],low_ind_start[1]:low_ind_end[1]]
            angle = angle[low_ind_start[0]:low_ind_end[0],low_ind_start[1]:low_ind_end[1]]

            # Actual subsampling
            lodft = lodft[:,low_ind_start[0]:low_ind_end[0],low_ind_start[1]:low_ind_end[1],:]

            # Filtering
            YIrcos = np.abs(np.sqrt(1 - Yrcos**2))
            lomask = pointOp(log_rad, YIrcos, Xrcos)
            lomask = torch.from_numpy(lomask[None,:,:,None]).float()
            lomask = lomask.to(self.device)

            # Convolution in spatial domain
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

        raise NotImplementedError('Reconstruction using PyTorch is work in progres...')

        if self.nbands != len(coeff[1]):
            raise Exception("Unmatched number of orientations")

        M, N = coeff[0].shape
        log_rad, angle = math_utils.utils.prepare_grid(M, N)

        Xrcos, Yrcos = math_utils.utils.rcosFn(1, -0.5)
        Yrcos  = np.sqrt(Yrcos)
        YIrcos = np.sqrt(np.abs(1 - Yrcos*Yrcos))

        lo0mask = pointOp(log_rad, YIrcos, Xrcos)
        hi0mask = pointOp(log_rad, Yrcos, Xrcos)

        tempdft = self._reconstruct_levels(coeff[1:], log_rad, Xrcos, Yrcos, angle)

        hidft = np.fft.fftshift(np.fft.fft2(coeff[0]))
        outdft = tempdft * lo0mask + hidft * hi0mask

        return np.fft.ifft2(np.fft.ifftshift(outdft)).real.astype(int)


    def _reconstruct_levels(self, coeff, log_rad, Xrcos, Yrcos, angle):
        
        raise NotImplementedError('Reconstruction using PyTorch is work in progres...')

        if len(coeff) == 1:
            # Single level remaining, just perform Fourier transform
            return np.fft.fftshift(np.fft.fft2(coeff[0]))

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
            orientdft += np.power(np.complex(0, 1), order) * banddft * anglemask * himask

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
        resdft = np.zeros(dims, 'complex')
        resdft[lostart[0]:loend[0], lostart[1]:loend[1]] = nresdft * lomask

        return resdft + orientdft

################################################################################
################################################################################
# Work in Progress

class TorchBatchedPyramid(object):

    def __init__(self, height, nbands):
        self._height = height  # including low-pass and high-pass
        self._nbands = nbands  # number of orientation bands
        self._coeff = [None]*self._nbands
    
    def set_level(self, level, coeff):
        assert len(coeff) == self._nbands
        assert coeff.shape[-1] == 2, 'Last dim must be 2 encoding real,imag'
        self._coeff[level] = coeff

    def level_size(self, level):
        if level in (0,self._nbands-1):
            # Low-pass and High-pass
            return self._coeff[level].shape[1:3]
        return self._coeff[level][0].shape[1:3]
