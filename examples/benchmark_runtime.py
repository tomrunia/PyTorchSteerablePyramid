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
# Date Created: 2018-12-10

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt

from steerable.SCFpyr_PyTorch import SCFpyr_PyTorch
from steerable.SCFpyr_NumPy import SCFpyr_NumPy
import steerable.utils as utils

import cortex.plot
cortex.plot.init_plotting()
colors = cortex.plot.nature_colors()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_file', type=str, default='./assets/patagonia.jpg')
    parser.add_argument('--batch_sizes', type=str, default='1,8,16,32,64,128,256')
    parser.add_argument('--image_sizes', type=str, default='128,256,512')
    parser.add_argument('--num_runs', type=int, default='5')
    parser.add_argument('--pyr_nlevels', type=int, default='5')
    parser.add_argument('--pyr_nbands', type=int, default='4')
    parser.add_argument('--pyr_scale_factor', type=int, default='2')
    parser.add_argument('--device', type=str, default='cuda:0')
    config = parser.parse_args()

    config.batch_sizes = list(map(int, config.batch_sizes.split(',')))
    config.image_sizes = list(map(int, config.image_sizes.split(',')))

    device = utils.get_device(config.device)

    ################################################################################

    pyr_numpy = SCFpyr_NumPy(
        height=config.pyr_nlevels, 
        nbands=config.pyr_nbands,
        scale_factor=config.pyr_scale_factor, 
    )

    pyr_torch = SCFpyr_PyTorch(
        height=config.pyr_nlevels, 
        nbands=config.pyr_nbands,
        scale_factor=config.pyr_scale_factor, 
        device=device
    )

    ############################################################################
    # Run Benchmark

    # durations_numpy = np.zeros((len(config.batch_sizes), len(config.image_sizes), config.num_runs))
    # durations_torch = np.zeros((len(config.batch_sizes), len(config.image_sizes), config.num_runs))
    
    # for batch_idx, batch_size in enumerate(config.batch_sizes):

    #     for size_idx, im_size in enumerate(config.image_sizes):

    #         for run_idx in range(config.num_runs):

    #             im_batch_numpy = utils.load_image_batch(config.image_file, batch_size, im_size)
    #             im_batch_torch = torch.from_numpy(im_batch_numpy).to(device)

    #             # NumPy implementation
    #             start_time = time.time()

    #             for image in im_batch_numpy:
    #                 coeff = pyr_numpy.build(image[0,])

    #             duration = time.time()-start_time
    #             durations_numpy[batch_idx,size_idx,run_idx] = duration
    #             print('BatchSize: {batch_size} | ImSize: {im_size} | NumPy Run {curr_run}/{num_run} | Duration: {duration:.3f} seconds.'.format(
    #                 batch_size=batch_size,
    #                 im_size=im_size,
    #                 curr_run=run_idx+1,
    #                 num_run=config.num_runs,
    #                 duration=duration
    #             ))

    #             # PyTorch Implementation
    #             start_time = time.time()

    #             im_batch_torch = torch.from_numpy(im_batch_numpy).to(device)
    #             coeff = pyr_torch.build(im_batch_torch)

    #             duration = time.time()-start_time
    #             durations_torch[batch_idx,size_idx,run_idx] = duration
    #             print('BatchSize: {batch_size} | ImSize: {im_size} | Torch Run {curr_run}/{num_run} | Duration: {duration:.3f} seconds.'.format(
    #                 batch_size=batch_size,
    #                 im_size=im_size,
    #                 curr_run=run_idx+1,
    #                 num_run=config.num_runs,
    #                 duration=duration
    #             ))

    # np.save('./assets/durations_numpy.npy', durations_numpy)
    # np.save('./assets/durations_torch.npy', durations_torch)
    
    ################################################################################
    # Plotting

    durations_numpy = np.load('./assets/durations_numpy.npy')
    durations_torch = np.load('./assets/durations_torch.npy')

    for i, num_examples in enumerate(config.batch_sizes):
        if num_examples == 8: continue
        avg_durations_numpy = np.mean(durations_numpy[i,:], -1) / num_examples
        avg_durations_torch = np.mean(durations_torch[i,:], -1) / num_examples
        plt.plot(config.image_sizes, avg_durations_numpy, marker='o', linestyle='-', lw=1.4, color=colors[i], label='numpy  (N = {})'.format(num_examples))
        plt.plot(config.image_sizes, avg_durations_torch, marker='d', linestyle='--', lw=1.4, color=colors[i], label='pytorch (N = {})'.format(num_examples))

    plt.title('Runtime Benchmark ({} levels, {} bands, average {numruns} runs)'.format(config.pyr_nlevels, config.pyr_nbands, numruns=config.num_runs))
    plt.xlabel('Image Size (px)')
    plt.ylabel('Time per Example (s)')
    plt.xlim((100, 550))
    plt.ylim((-0.01, 0.2))
    plt.xticks(config.image_sizes)
    plt.legend(ncol=2, loc='top left')
    plt.tight_layout()
    
    plt.savefig('./assets/runtime_benchmark.png', dpi=600)
    plt.savefig('./assets/runtime_benchmark.pdf')
    plt.show()
