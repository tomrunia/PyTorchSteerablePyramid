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


from steerable.SCFpyr import SCFpyr
from common.utils import visualize

import cv2


if __name__ == "__main__":

    image_file = './assets/lena.jpg'
    im = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    im = cv2.resize(im, dsize=(200, 200))

    # Build the complex steerable pyramid
    pyr = SCFpyr(height=5)
    coeff = pyr.build(im)

    # Visualization of whole decomposition
    cv2.imshow('coeff', visualize(coeff))

    # reconstruction
    out = pyr.reconSCFpyr(coeff)

    cv2.imshow('sub', coeff[1][0].real)
    cv2.imshow('image', im)
    cv2.imshow('reconstruction', out*255)
    cv2.waitKey(0)
    