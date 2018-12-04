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
################################################################################

def visualize(coeff, normalize=True):
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
