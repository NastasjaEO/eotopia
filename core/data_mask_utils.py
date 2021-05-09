# -*- coding: utf-8 -*-
"""
Created on Sat May  8 15:40:40 2021

@author: freeridingeo
"""

import numpy as np


def negate_mask(mask):
    """Returns the negated mask.

    If elements of input mask have 0 and non-zero values, then the 
    returned matrix will have all elements 0 (1) where
    the original one has non-zero (0).
    
    :param mask: Input mask
    :type mask: np.array
    :return: array of same shape and dtype=int8 as input array
    :rtype: np.array
    """
    res = np.ones(mask.shape, dtype=np.int8)
    res[mask > 0] = 0

    return res

