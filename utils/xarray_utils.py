# -*- coding: utf-8 -*-
"""
Created on Fri May 14 14:11:23 2021

@author: freeridingeo
"""

import xarray as xr

import sys
sys.path.append("D:/Code/eotopia/utils")
from dataframe_utils import feature_array_to_dataframe

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



