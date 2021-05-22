# -*- coding: utf-8 -*-
"""
Created on Sat May  8 15:40:40 2021

@author: freeridingeo
"""

import numpy as np
from eolearn.core import EOPatch, FeatureType

def mask_data_by_FeatureMask(eopatch, data_da, mask):
    """
    Creates a copy of array and insert 0 where data is masked.
        
    :param data_da: dataarray
    :type data_da: xarray.DataArray
    :return: dataaray
    :rtype: xarray.DataArray
    """
    mask = eopatch[FeatureType.MASK][mask]
    if len(data_da.values.shape) == 4:
        mask = np.repeat(mask, data_da.values.shape[-1], -1)
    else:
        mask = np.squeeze(mask, axis=-1)
    data_da = data_da.copy()
    data_da.values[~mask] = 0
    return data_da

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

