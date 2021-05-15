# -*- coding: utf-8 -*-
"""
Created on Fri May 14 11:37:27 2021

@author: freeridingeo
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from sentinelhub import CRS
from cartopy import crs as ccrs

from eolearn.core import EOPatch, FeatureType, FeatureTypeSet, FeatureParser

import geoviews as gv
import holoviews as hv

import sys
sys.path.append("D:/Code/eotopia/core")
from data_bands_utils import mask_data_by_FeatureMask

sys.path.append("D:/Code/eotopia/utils")
from string_utils import string_to_variable
from dataframe_utils import feature_array_to_dataframe
from coordinate_utils import new_xarray_coordinates
from xarray_utils import eopatch_dataframe_to_rasterband

sys.path.append("D:/Code/eotopia/metadata")
from metadata_temporal import fill_vector_of_eopatch_timestamps

PLOT_WIDTH = 800
PLOT_HEIGHT = 500

patchpath="D:/Code/eotopia/tests/testdata/TestEOPatch"
eopatch = EOPatch.load(patchpath)

features = eopatch.get_feature_list()
feature = eopatch.get_feature(FeatureType.BBOX)

feature_list = list(FeatureParser(features))

feature_type, feature_name = feature_list[0]
print(feature_type)
print(feature_name)


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
    vis = data_da.hvplot(x='x', y='y',
                             crs=ccrs.epsg(crs.epsg))\
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

def plot_eopatch_data(eopatch, feature_type, feature_name):
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
    data_band = eopatch_dataframe_to_rasterband(data_da,feature_name, crs)
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


