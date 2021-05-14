# -*- coding: utf-8 -*-
"""
Created on Fri May 14 14:11:23 2021

@author: freeridingeo
"""

import numpy as np
import xarray as xr

from eolearn.core import FeatureParser

from .metadata_geographical import (get_eopatch_coordinates)
from .metadata_eopatch import get_feature_dimensions

from .utils.string_utils import string_to_variable

def feature_array_to_dataframe(eopatch, feature, remove_depth=True, crs=None, convert_bool=True):
    """ 
    Converts one feature of eopatch to xarray DataArray
    
    :param eopatch: eopatch
    :type eopatch: EOPatch
    :param feature: feature of eopatch
    :type feature: (FeatureType, str)
    :param remove_depth: removes last dimension if it is one
    :type remove_depth: bool
    :param crs: converts dimensions to crs
    :type crs: sentinelhub.crs
    :param convert_bool: If True it will convert boolean dtype into uint8 dtype
    :type convert_bool: bool
    :return: dataarray
    :rtype: xarray DataArray
    """
    features = list(FeatureParser(feature))
    feature_type, feature_name = features[0]
    bbox = eopatch.bbox
    data = eopatch[feature_type][feature_name]
    if isinstance(data, xr.DataArray):
        data = data.values
    dimensions = get_feature_dimensions(feature)
    coordinates = get_eopatch_coordinates(eopatch, feature, crs=crs)
    dataframe = xr.DataArray(data=data,
                             coords=coordinates,
                             dims=dimensions,
                             attrs={'crs': str(bbox.crs),
                                    'feature_type': feature_type,
                                    'feature_name': feature_name},
                             name=string_to_variable(feature_name))

    if remove_depth and dataframe.values.shape[-1] == 1:
        dataframe = dataframe.squeeze()
        dataframe = dataframe.drop(feature_name + '_dim')
    if convert_bool and dataframe.dtype == bool:
        dataframe = dataframe.astype(np.uint8)
    return dataframe



