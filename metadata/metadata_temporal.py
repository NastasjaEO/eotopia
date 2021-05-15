# -*- coding: utf-8 -*-
"""
Created on Fri May 14 13:42:14 2021

@author: freeridingeo
"""

import pandas as pd
import geopandas as gpd

import sys
sys.path.append("D:/Code/eotopia/utils")
from dataframe_utils import create_dummy_dataframe_for_blank_timestamps
from geometry_utils import create_dummy_polygon

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

def fill_vector_of_eopatch_timestamps(eopatch, feature_type, feature_name,
                                      timestamp_column='TIMESTAMP', 
                                      geometry_column='geometry'):
    """ 
    Adds timestamps from eopatch to GeoDataFrame.
        
    :param feature_type: type of eopatch feature
    :type feature_type: FeatureType
    :param feature_name: name of eopatch feature
    :type feature_name: str
    :param timestamp_column: geopandas.GeoDataFrame columns with timestamps
    :type timestamp_column: str
    :return: GeoDataFrame with added data
    :rtype: geopandas.GeoDataFrame
    """
    vector = eopatch[feature_type][feature_name].copy()
    vector['valid'] = True
    eopatch_timestamps = eopatch.timestamp
    vector_timestamps = set(vector[timestamp_column])
    blank_timestamps = [timestamp\
                            for timestamp in eopatch_timestamps\
                                if timestamp not in vector_timestamps]
    dummy_geometry = create_dummy_polygon(0.0000001)
    temp_df = create_dummy_dataframe_for_blank_timestamps(vector,
                                     blank_timestamps=blank_timestamps,
                                     dummy_geometry=dummy_geometry)

    final_vector = gpd.GeoDataFrame(pd.concat((vector, temp_df), 
                                                  ignore_index=True),
                                                    crs=vector.crs)
    return final_vector


