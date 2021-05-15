# -*- coding: utf-8 -*-
"""
Created on Fri May 14 14:11:23 2021

@author: freeridingeo
"""

import numpy as np
import xarray as xr
from sentinelhub import CRS

import sys
sys.path.append("D:/Code/eotopia/utils")
from dataframe_utils import feature_array_to_dataframe
from string_utils import string_to_variable
from coordinate_utils import new_xarray_coordinates

def eopatch_to_xrdataset(eopatch, remove_depth=True):
    """
    Converts eopatch to xarray Dataset

    :param eopatch: eopathc
    :type eopatch: EOPatch
    :param remove_depth: removes last dimension if it is one
    :type remove_depth: bool
    :return: dataset
    :rtype: xarray Dataset
    """
    dataset = xr.Dataset()
    for feature in eopatch.get_feature_list():
        if not isinstance(feature, tuple):
            continue
        feature_type = feature[0]
        feature_name = feature[1]
        if feature_type.is_raster():
            dataframe = feature_array_to_dataframe(eopatch, 
                                                   (feature_type, 
                                                    feature_name), 
                                                   remove_depth)
            dataset[feature_name] = dataframe
    return dataset

def eopatch_dataframe_to_rasterband(eopatch_da, feature_name, crs, band,
                                    rgb_factor=None):
    """ 
    Creates new xarray DataArray (from old one)
        
    :param eopatch_da: eopatch DataArray
    :type eopatch_da: DataArray
    :param feature_name: name of the feature to plot
    :type feature_name:  str
    :param crs: in which crs are the data
    :type crs: sentinelhub.constants.crs
    :return: eopatch DataArray with proper coordinates, dimensions, crs
    :rtype: xarray.DataArray
    """
    timestamps = eopatch_da.coords['time'].values
    band = list(band) if isinstance(band, tuple) else band
    if rgb_factor:    
        bands = eopatch_da[..., band] * rgb_factor
    else:
        bands = eopatch_da[..., band]

    bands = bands.rename({string_to_variable(feature_name, '_dim'): 'band'})\
        .transpose('time', 'band', 'y', 'x')
    x_values, y_values = new_xarray_coordinates(eopatch_da, crs, CRS.POP_WEB)
    eopatch_band = xr.DataArray(data=np.clip(bands.data, 0, 1),
                                   coords={'time': timestamps,
                                           'band': band,
                                           'y': np.flip(y_values),
                                           'x': x_values},
                                   dims=('time', 'band', 'y', 'x'))
    return eopatch_band



