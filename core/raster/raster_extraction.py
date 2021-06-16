# -*- coding: utf-8 -*-
"""
Created on Wed May 26 10:30:52 2021

@author: freeridingeo
"""

import numpy as np
import rasterio


def get_pixelvalue_at_latlongpoint(rasterpath, long, lat, iband=1):
    """
    Return pixel coordinates of raster image corresponding on the coordinates
    of a point.
    """   
    with rasterio.open(rasterpath) as src:        # Get image and params.
        img = src.read(iband)
        width, height = img.shape
        x = np.array([long])
        y = np.array([lat])       
        idxs = [src.index(point[0], point[1]) for point in zip(x, y)]
    return idxs

def return_pixel_index(pixel_coordinates, window):
    """
    pixel_coordinates: Tuple
    window: Window
    Return indexes of pixel mapping for raster window.
    """
    (row_min, row_max), (col_min, col_max) = window.toranges()

    index_window = np.logical_and.reduce(
        (
            pixel_coordinates[0] >= row_min,
            pixel_coordinates[0] < row_max,
            pixel_coordinates[1] >= col_min,
            pixel_coordinates[1] < col_max,
        )
    )
    return index_window

def extract_pixel_value(pixel_coordinates, window):
    """
    pixel_coordinates: Tuple
    window: Window
    """
    index_window = return_pixel_index(pixel_coordinates, window)

    (row_min, row_max), (col_min, col_max) = window.toranges()
    pixel_mapping_window = (
        pixel_coordinates[0][index_window] - row_min,
        pixel_coordinates[1][index_window] - col_min,
    )
    return pixel_mapping_window, index_window