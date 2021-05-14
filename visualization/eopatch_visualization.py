# -*- coding: utf-8 -*-
"""
Created on Fri May 14 11:37:27 2021

@author: freeridingeo
"""

from sentinelhub import CRS

from eolearn.core import FeatureType, FeatureTypeSet, FeatureParser

import geoviews as gv


class EOPatchVisualization:
    """
    Plot class for making visulizations.

    :param eopatch: eopatch
    :type eopatch: EOPatch
    :param feature: feature of eopatch
    :type feature: (FeatureType, str)
    :param rgb: bands for creating RGB image
    :type rgb: [int, int, int]
    :param rgb_factor: multiplication factor for constructing rgb image
    :type rgb_factor: float
    :param vdims: value dimensions for plotting geopandas.GeoDataFrame
    :type vdims: str
    :param timestamp_column: geopandas.GeoDataFrame columns with timestamps
    :type timestamp_column: str
    :param geometry_column: geopandas.GeoDataFrame columns with geometry
    :type geometry_column: geometry
    :param pixel: wheather plot data for each pixel (line), for FeatureType.DATA and FeatureType.MASK
    :type pixel: bool
    :param mask: name of the FeatureType.MASK to apply to data
    :type mask: str
    
    """
    def __init__(self, eopatch, feature, rgb=None, rgb_factor=3.5, vdims=None,
                 timestamp_column='TIMESTAMP', geometry_column='geometry', pixel=False, mask=None):
        self.eopatch = eopatch
        self.feature = feature
        self.rgb = list(rgb) if isinstance(rgb, tuple) else rgb
        self.rgb_factor = rgb_factor
        self.vdims = vdims
        self.timestamp_column = timestamp_column
        self.geometry_column = geometry_column
        self.pixel = pixel
        self.mask = mask

    def plot(self):
        features = list(FeatureParser(self.feature))
        feature_type, feature_name = features[0]

    def plot_data(self, feature_name):
        """ 
        Plots the FeatureType.DATA of eopatch.

        :param feature_name: name of the eopatch feature
        :type feature_name: str
        :return: visualization
        :rtype: holoview/geoviews/bokeh
        """
        crs = self.eopatch.bbox.crs
        crs = CRS.POP_WEB if crs is CRS.WGS84 else crs

