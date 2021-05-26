# -*- coding: utf-8 -*-
"""
Created on Wed May 26 10:31:52 2021

@author: freeridingeo
"""

import rasterio

def raster_bandvalue_histogram(raster, bins=50):
    rasterio.plot.show_hist(raster, bins=bins, lw=0.0, 
                            stacked=False, alpha=0.3,
                            histtype='stepfilled', title="Histogram")
