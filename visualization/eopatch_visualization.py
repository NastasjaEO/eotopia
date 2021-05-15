# -*- coding: utf-8 -*-
"""
Created on Fri May 14 11:37:27 2021

@author: freeridingeo
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from sentinelhub import CRS
from shapely import speedups
speedups.disable()
from cartopy import crs as ccrs

from eolearn.core import FeatureType

import geoviews as gv
import holoviews as hv
import hvplot.xarray

import sys
sys.path.append("D:/Code/eotopia/core")
from data_mask_utils import mask_data_by_FeatureMask

sys.path.append("D:/Code/eotopia/utils")
from dataframe_utils import feature_array_to_dataframe
from xarray_utils import eopatch_dataframe_to_rasterband

sys.path.append("D:/Code/eotopia/utils")
from geometry_utils import create_dummy_polygon


PLOT_WIDTH = 800
PLOT_HEIGHT = 500

def plot_eopatch_raster(eopatch, feature_type, feature_name):
    """
    Makes visualization for raster data (except for FeatureType.DATA)

    :param feature_type: type of eopatch feature
    :type feature_type: FeatureType
    :param feature_name: name of eopatch feature
    :type feature_name: str
    :return: visualization
    :rtype: holoviews/geoviews/bokeh
    """
    crs = eopatch.bbox.crs
    crs = CRS.POP_WEB if crs is CRS.WGS84 else crs
    data_da = feature_array_to_dataframe(eopatch, 
                                             (feature_type, 
                                              feature_name), crs=crs)
    data_min = data_da.values.min()
    data_max = data_da.values.max()
    data_levels = len(np.unique(data_da))
    data_levels = 11 if data_levels > 11 else data_levels
    data_da = data_da.where(data_da > 0).fillna(-1)
    
    data = eopatch[feature_type, feature_name]
    vis = data_da.hvplot(x='x', y='y', crs=ccrs.epsg(crs.epsg))\
                .opts(clim=(data_min, data_max),
                      clipping_colors={'min': 'transparent'},
                      color_levels=data_levels)
    return vis.opts(plot=dict(width=PLOT_WIDTH, height=PLOT_HEIGHT))

def plot_one_band(eopatch_da, timestamp):  # OK
    """ 
    Returns visualization for one timestamp for FeatureType.DATA
        
    :param eopatch_da: eopatch converted to xarray DataArray
    :type eopatch_da: xarray DataArray
    :param timestamp: timestamp to make plot for
    :type timestamp: datetime
    :return: visualization
    :rtype:  holoviews/geoviews/bokeh
    """
    return eopatch_da.sel(time=timestamp).drop('time').hvplot(x='x', y='y')

def plot_eopatch_data(eopatch, feature_type, feature_name, band=1):
    """
    Plots the FeatureType.DATA of eopatch.

    :param feature_name: name of the eopatch feature
    :type feature_name: str
    :return: visualization
    :rtype: holoview/geoviews/bokeh
    """
    crs = eopatch.bbox.crs
    crs = CRS.POP_WEB if crs is CRS.WGS84 else crs
    data_da = feature_array_to_dataframe(eopatch, 
                                             (FeatureType.DATA, 
                                              feature_name), crs=crs)
    timestamps = eopatch.timestamp
    crs = eopatch.bbox.crs
    data_band = eopatch_dataframe_to_rasterband(data_da,feature_name, crs, band)
    band_dict = {timestamp_: plot_one_band(data_band, timestamp_)\
                    for timestamp_ in timestamps}
    vis = hv.HoloMap(band_dict , kdims=['time'])
    return vis.opts(plot=dict(width=PLOT_WIDTH, height=PLOT_HEIGHT))

def plot_shapes_one(data_gpd, timestamp, crs, timestamp_column='TIMESTAMP'):
    """
    Plots shapes for one timestamp from geopandas GeoDataFrame
        
    :param data_gpd: data to plot
    :type data_gpd: geopandas.GeoDataFrame
    :param timestamp: timestamp to plot data for
    :type timestamp: datetime
    :param crs: in which crs is the data to plot
    :type crs: sentinelhub.crs
    :return: visualization
    :rtype: geoviews
    """
    out = data_gpd.loc[data_gpd[timestamp_column] == timestamp]
    return gv.Polygons(out, crs=ccrs.epsg(int(crs.value)))

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

def plot_vector(eopatch, feature_name):
    """ 
    Visualizaton for vector (FeatureType.VECTOR) data

    :param feature_name: name of eopatch feature
    :type feature_name: str
    :return: visualization
    :rtype: holoviews/geoviews/bokeh
    """
    crs = eopatch.bbox.crs
    timestamps = eopatch.timestamp
    data_gpd = fill_vector_of_eopatch_timestamps(FeatureType.VECTOR, 
                                                 feature_name)
    if crs is CRS.WGS84:
        crs = CRS.POP_WEB
        data_gpd = data_gpd.to_crs(crs.pyproj_crs())
    shapes_dict = {timestamp_: plot_shapes_one(data_gpd, timestamp_, crs)
                       for timestamp_ in timestamps}
    vis = hv.HoloMap(shapes_dict, kdims=['time'])
    return vis.opts(plot=dict(width=PLOT_WIDTH, height=PLOT_HEIGHT))

def plot_scalar_label(eopatch, feature_type, feature_name):
    """ 
    Line plot for FeatureType.SCALAR, FeatureType.LABEL

    :param feature_type: type of eopatch feature
    :type feature_type: FeatureType
    :param feature_name: name of eopatch feature
    :type feature_name: str
    :return: visualization
    :rtype: holoviews/geoviews/bokeh
    """
    data_da = feature_array_to_dataframe(eopatch, (feature_type, feature_name))
    vis = data_da.hvplot()
    return vis.opts(plot=dict(width=PLOT_WIDTH, height=PLOT_HEIGHT))

def plot_vector_timeless(eopatch, feature_name, vdims=None):
    """ 
    Plot FeatureType.VECTOR_TIMELESS data
        
    :param feature_name: name of the eopatch featrue
    :type feature_name: str
    :return: visalization
    :rtype: geoviews
    """
    crs = eopatch.bbox.crs
    data_gpd = eopatch[FeatureType.VECTOR_TIMELESS][feature_name]
    if crs is CRS.WGS84:
        crs = CRS.POP_WEB
        data_gpd = data_gpd.to_crs(crs.pyproj_crs())
    vis = gv.Polygons(data_gpd, crs=ccrs.epsg(crs.epsg), vdims=vdims)
    return vis.opts(plot=dict(width=PLOT_WIDTH, height=PLOT_HEIGHT))

def plot_pixel(eoptach, pixel, feature_type, feature_name, mask=None):
    """
    Plots one pixel through time
        
    :return: visualization
    :rtype: holoviews
    """
    data_da = feature_array_to_dataframe(eopatch, (feature_type, feature_name))
    if mask:
        data_da = mask_data_by_FeatureMask(eopatch, data_da, mask)
    vis = data_da.hvplot(x='time')
    return vis.opts(plot=dict(width=PLOT_WIDTH, height=PLOT_HEIGHT))


