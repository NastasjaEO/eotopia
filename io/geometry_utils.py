# -*- coding: utf-8 -*-
"""
Created on Fri May 21 11:07:09 2021

@author: freeridingeo
"""

import rasterio
from sentinelhub import BBox, bbox_to_dimensions, CRS

def raster_size_from_aoi_bbox(list_of_coords, resolution):
    """
    Example for list_of_coords: [13.353882, 45.402307, 16.644287, 46.908998]
    """
    aoi_bbox = BBox(list_of_coords, crs=CRS.WGS84)
    aoi_size = bbox_to_dimensions(aoi_bbox, resolution)
    return aoi_size

def get_boundaries_around_singlepixelcoordinate(rasterpath, 
                                                      lat, long, 
                                                      num_pixels,
                                                      aspixel="False"):
    with rasterio.open(rasterpath, "r") as src:
        row0, col0 = src.index(long, lat)

    delta = int(num_pixels/2)
    row_lower = row0 - delta
    row_upper = row0 + delta
    col_lower = col0 - delta
    col_upper = col0 + delta

    if aspixel == "True":
        row_lower, col_lower = src.xy(row_lower, col_lower)
        row_upper, col_upper = src.xy(row_upper, col_upper)

    return row_lower, row_upper, col_lower, col_upper

def window_to_index(wind_):
    """
    Generates a list of index (row,col): [[row1,col1],[row2,col2],[row3,col3],[row4,col4],[row1,col1]]
    pol_index = window_to_index(window_slice)
    """
    return [[wind_.row_off,wind_.col_off],
            [wind_.row_off,wind_.col_off+wind_.width],
            [wind_.row_off+wind_.height,wind_.col_off+wind_.width],
            [wind_.row_off+wind_.height,wind_.col_off],
            [wind_.row_off,wind_.col_off]]


