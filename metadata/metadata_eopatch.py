# -*- coding: utf-8 -*-
"""
Created on Wed May 12 14:07:28 2021

@author: freeridingeo
"""

import numpy as np

from eolearn.core import FeatureTypeSet, FeatureParser

from .utils.string_utils import string_to_variable

def _get_eopatch_depth_coordinates(feature_name, data, names_of_channels=None):
    """ 
    Returns band/channel/dept coordinates for xarray DataArray/Dataset
    
    :param feature_name: name of feature of EOPatch
    :type feature_name: FeatureType
    :param data: data of EOPatch
    :type data: numpy.array
    :param names_of_channels: coordinates for the last (band/dept/chanel) dimension
    :type names_of_channels: list
    :return: depth/band coordinates
    :rtype: dict
    """
    coordinates = {}
    depth = string_to_variable(feature_name, '_dim')
    if names_of_channels:
        coordinates[depth] = names_of_channels
    elif isinstance(data, np.ndarray):
        coordinates[depth] = np.arange(data.shape[-1])
    return coordinates

def get_feature_dimensions(feature):
    """ 
    Returns list of dimensions for xarray DataArray/Dataset
    
    :param feature: eopatch feature
    :type feature: (FeatureType, str)
    :return: dimensions for xarray DataArray/Dataset
    :rtype: list(str)
    """
    features = list(FeatureParser(feature))
    feature_type, feature_name = features[0]
    depth = string_to_variable(feature_name, '_dim')
    if feature_type in FeatureTypeSet.RASTER_TYPES_4D:
        return ['time', 'y', 'x', depth]
    if feature_type in FeatureTypeSet.RASTER_TYPES_2D:
        return ['time', depth]
    if feature_type in FeatureTypeSet.RASTER_TYPES_3D:
        return ['y', 'x', depth]
    return [depth]

def add_metadata_to_eopatch(eopatch, metainfo):
    eopatch.meta_info[str(metainfo)] = metainfo
    return eopatch

def get_eopatch_size(eopatch):
    return eopatch.meta_info.get('size_x'), eopatch.meta_info.get('size_y')

def get_eopatch_bbox(eopatch):
    return eopatch.bbox

