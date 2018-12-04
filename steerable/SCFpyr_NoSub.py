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
import scipy.misc as sc
import scipy.signal

from steerable.SCFpyr import SCFpyr

################################################################################
################################################################################

class SCFpyrNoSub(SCFpyr):

	def buildSCFpyrlevs(self, lodft, log_rad, angle, Xrcos, Yrcos, ht):
		if (ht <= 1):
			lo0 = np.fft.ifft2(np.fft.ifftshift(lodft))
			coeff = [lo0.real]

		else:
			Xrcos = Xrcos - 1

			# ==================== Orientation bandpass =======================
			himask = self.pointOp(log_rad, Yrcos, Xrcos)

			lutsize = 1024
			Xcosn = np.pi * np.array(range(-(2*lutsize+1), (lutsize+2)))/lutsize
			order = self.nbands - 1
			const = np.power(2, 2*order) * np.square(sc.factorial(order)
                                            ) / (self.nbands * sc.factorial(2*order))

			alpha = (Xcosn + np.pi) % (2*np.pi) - np.pi
			Ycosn = 2*np.sqrt(const) * np.power(np.cos(Xcosn),
                                       order) * (np.abs(alpha) < np.pi/2)

			orients = []

			for b in range(self.nbands):
				anglemask = self.pointOp(angle, Ycosn, Xcosn + np.pi*b/self.nbands)
				banddft = np.power(np.complex(0, -1), self.nbands - 1) * \
                                    lodft * anglemask * himask
				band = np.fft.ifft2(np.fft.ifftshift(banddft))
				orients.append(band)

			# ================== Subsample lowpass ============================
			lostart = (0, 0)
			loend = lodft.shape

			log_rad = log_rad[lostart[0]:loend[0], lostart[1]:loend[1]]
			angle = angle[lostart[0]:loend[0], lostart[1]:loend[1]]
			lodft = lodft[lostart[0]:loend[0], lostart[1]:loend[1]]
			YIrcos = np.abs(np.sqrt(1 - Yrcos*Yrcos))
			lomask = self.pointOp(log_rad, YIrcos, Xrcos)

			lodft = lomask * lodft

			coeff = self.buildSCFpyrlevs(lodft, log_rad, angle, Xrcos, Yrcos, ht-1)
			coeff.insert(0, orients)

		return coeff
