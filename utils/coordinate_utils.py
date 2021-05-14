# -*- coding: utf-8 -*-
"""
Created on Fri May 14 14:20:30 2021

@author: freeridingeo
"""

import numpy as np
from sentinelhub import BBox


def new_xarray_coordinates(data, crs, new_crs):
    """ 
    Returns coordinates for xarray DataArray/Dataset in new crs.
    
    :param data: data for converting coordinates for
    :type data: xarray.DataArray or xarray.Dataset
    :param crs: old crs
    :type crs: sentinelhub.CRS
    :param new_crs: new crs
    :type new_crs: sentinelhub.CRS
    :return: new x and y coordinates
    :rtype: (float, float)
    """
    x_values = data.coords['x'].values
    y_values = data.coords['y'].values
    bbox = BBox((x_values[0], y_values[0], x_values[-1], y_values[-1]), crs=crs)
    bbox = bbox.transform(new_crs)
    xmin, ymin = bbox.lower_left
    xmax, ymax = bbox.upper_right
    new_xs = np.linspace(xmin, xmax, len(x_values))
    new_ys = np.linspace(ymin, ymax, len(y_values))

    return new_xs, new_ys