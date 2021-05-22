# -*- coding: utf-8 -*-
"""
Created on Fri May 14 13:37:46 2021

@author: freeridingeo
"""

import numpy as np

from eolearn.core import FeatureTypeSet, FeatureParser

import sys
sys.path.append("D:/Code/eotopia/metadata")
from metadata_temporal import _return_timestamps_of_eopatch
from metadata_eopatch import _get_eopatch_depth_coordinates

def get_eopatch_bbox(eopatch):
    return eopatch.bbox

def _get_spatial_coordinates_of_eopatch(bbox, data, feature_type):
    """ 
    Returns spatial coordinates (dictionary) for creating 
    xarray DataArray/Dataset
    Makes sense for data

    :param bbox: eopatch bbox
    :type bbox: EOPatch BBox
    :param data: values for calculating number of coordinates
    :type data: numpy array
    :param feature_type: type of the feature
    :type feature_type: FeatureType
    :return: spatial coordinates
    :rtype: dict {'x':, 'y':}
    """ 

    if not (feature_type.is_spatial() and feature_type.is_raster()):
        raise ValueError('Data should be raster and have spatial dimension')
    index_x, index_y = 2, 1
    if feature_type.is_timeless():
        index_x, index_y = 1, 0
    pixel_width = (bbox.max_x - bbox.min_x)/data.shape[index_x]
    pixel_height = (bbox.max_y - bbox.min_y)/data.shape[index_y]
    return {'x': np.linspace(bbox.min_x+pixel_width/2, bbox.max_x-pixel_width/2, data.shape[index_x]),
            'y': np.linspace(bbox.max_y-pixel_height/2, bbox.min_y+pixel_height/2, data.shape[index_y])}

def get_eopatch_coordinates(eopatch, feature, crs):
    """ 
    Creates coordinates for xarray DataArray
    
    :param eopatch: eopatch
    :type eopatch: EOPatch
    :param feature: feature of eopatch
    :type feature: (FeatureType, str)
    :param crs: convert spatial coordinates to crs
    :type crs: sentinelhub.crs
    :return: coordinates for xarry DataArray/Dataset
    :rtype: dict
    """
    features = list(FeatureParser(feature))
    feature_type, feature_name = features[0]
    original_crs = eopatch.bbox.crs
    if crs and original_crs != crs:
        bbox = eopatch.bbox.transform(crs)
    else:
        bbox = eopatch.bbox
    data = eopatch[feature_type][feature_name]
    timestamps = eopatch.timestamp

    if feature_type in FeatureTypeSet.RASTER_TYPES_4D:
        return {**_return_timestamps_of_eopatch(timestamps),
                **_get_spatial_coordinates_of_eopatch(bbox, data, 
                                                      feature_type),
                **_get_eopatch_depth_coordinates(data=data, 
                                                 feature_name=feature_name)}
    if feature_type in FeatureTypeSet.RASTER_TYPES_2D:
        return {**_return_timestamps_of_eopatch(timestamps),
                **_get_eopatch_depth_coordinates(data=data, 
                                                 feature_name=feature_name)}
    if feature_type in FeatureTypeSet.RASTER_TYPES_3D:
        return {**_get_spatial_coordinates_of_eopatch(bbox, data, 
                                                      feature_type),
                **_get_eopatch_depth_coordinates(data=data, 
                                                 feature_name=feature_name)}
    return _get_eopatch_depth_coordinates(data=data, 
                                          feature_name=feature_name)

    


