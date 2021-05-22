# -*- coding: utf-8 -*-
"""
Created on Fri May 14 14:11:23 2021

@author: freeridingeo
"""

import numpy as np
import xarray as xr
import pandas as pd

from eolearn.core import FeatureParser

import sys
sys.path.append("D:/Code/eotopia/metadata")
from metadata_geographical import (get_eopatch_coordinates)
from metadata_eopatch import get_feature_dimensions

sys.path.append("D:/Code/eotopia/utils")
from string_utils import string_to_variable

import matplotlib.pyplot as plt

def filter_nan(s,o):
    data = np.transpose(np.array([s.flatten(),o.flatten()]))
    data = data[~np.isnan(data).any(1)]
    return data[:,0], data[:,1]

def extract_difference_between_columns(df1, df2, col1, col2):
    common = df1.merge(df2,on=[col1,col2])
    diff = df1[(~df1[col1].isin(common[col1])) & (~df1[col2].isin(common[col2]))]
    return diff

def df_grouby_and_count(df, col):
    group_by = df.groupby(by=[col])
    col_avg = group_by.mean()
    col_count = group_by.count()
    print("Mean value of " + str(col) + " is ", col_avg)
    print("Number of " + str(col) + " is ", col_count)

def concatenate_dfs(list_of_dfs, kind="by_append"):
    if kind == "by_append":
        df_conc = pd.concat(list_of_dfs)
    elif kind == "matrixed":
        df_conc = pd.concat(list_of_dfs, axis=1)
    return df_conc

def merge_dfs(df1, df2, colname, kind="inner"):
    """
    how:    Options: 
                "inner": print for common rows
                "outer": print for all rows, not just common rows
                "df1/df2"
    """
    merged_df = pd.merge(df1, df2, how=kind, on=colname)
    return merged_df

def join_dfs(df1, df2, kind="inner"):
    """
    Joining is a convenient method for combining the columns of two potentially 
    differently-indexed DataFrames into a single result DataFrame.
    how:    Options: 
                "inner": print for common rows
                "outer": print for all rows, not just common rows
    """
    join_df = pd.merge(df1, df2, how=kind)
    return join_df

def identify_unique(dataframe):
    unique_counts = dataframe.nunique()
    unique_stats = pd.DataFrame(unique_counts).rename(columns = {'index': 'feature', 0: 'nunique'})
    unique_stats = unique_stats.sort_values('nunique', ascending = True)

    # Find the columns with only one unique count
    uniques = pd.DataFrame(unique_counts[unique_counts == 1]).reset_index().rename(columns = {'index': 'feature', 0: 'nunique'})
    unique_stats.plot.hist(edgecolor = 'k', figsize = (7, 5))
    plt.ylabel('Frequency', size = 14); plt.xlabel('Unique Values', size = 14); 
    plt.title('Number of Unique Values Histogram', size = 16);

    return uniques

def feature_array_to_dataframe(eopatch, feature, remove_depth=True, 
                               crs=None, convert_bool=True):
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

def create_dummy_dataframe_for_blank_timestamps(geodataframe,
                                                blank_timestamps, dummy_geometry,
                                                timestamp_column='TIMESTAMP', 
                                                geometry_column='geometry',
                                                fill_str='', fill_numeric=1):
    """ 
    Creates gpd GeoDataFrame to fill with dummy data 
        
    :param geodataframe: dataframe to append rows to
    :type geodataframe: geopandas.GeoDataFrame
    :param timestamp_column: geopandas.GeoDataFrame columns with timestamps
    :type timestamp_column: str
    :param blank_timestamps: timestamps for constructing dataframe
    :type blank_timestamps: list of timestamps
    :param dummy_geometry: geometry to plot when there is no data
    :type dummy_geometry: shapely.geometry.Polygon
    :param fill_str: insert when there is no value in str column
    :type fill_str: str
    :param fill_numeric: insert when
    :type fill_numeric: float
    :return: dataframe with dummy data
    :rtype: geopandas.GeoDataFrame
    """
    dataframe = pd.DataFrame(data=blank_timestamps, columns=[timestamp_column])

    for column in geodataframe.columns:
        if column == timestamp_column:
            continue
        if column == geometry_column:
            dataframe[column] = dummy_geometry
        elif column == 'valid':
            dataframe[column] = False
        elif geodataframe[column].dtype in (int, float):
            dataframe[column] = fill_numeric
        else:
            dataframe[column] = fill_str
    return dataframe

