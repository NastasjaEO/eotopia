# -*- coding: utf-8 -*-
"""
Created on Fri May 14 13:42:14 2021

@author: freeridingeo
"""

import numpy as np
import xarray as xr

def _return_timestamps_of_eopatch(timestamps):
    """ 
    Returns temporal coordinates dictionary for creating 
    xarray DataArray/Dataset
    
    :param timestamps: timestamps
    :type timestamps: EOpatch.timestamp
    :return: temporal coordinates
    :rtype: dict {'time': }
    """
    return {'time': timestamps}


