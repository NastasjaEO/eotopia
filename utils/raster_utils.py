# -*- coding: utf-8 -*-
"""
Created on Thu May 13 15:15:23 2021

@author: freeridingeo
"""

import os
import numpy as np
import rasterio

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

def raster_bandvalue_histogram(raster, bins=50):
    rasterio.plot.show_hist(raster, bins=bins, lw=0.0, 
                            stacked=False, alpha=0.3,
                            histtype='stepfilled', title="Histogram")


