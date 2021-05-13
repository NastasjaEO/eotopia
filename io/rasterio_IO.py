# -*- coding: utf-8 -*-
"""
Created on Thu May 13 11:40:34 2021

@author: freeridingeo
"""

from pathlib import Path
import rasterio
from rasterio.plot import show
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from rasterio.crs import CRS

import matplotlib.pyplot as plt

def read_raster_by_bbox(rasterpath, bbox, crs=4326, plot=False):
    """
    rasterpath: Path to raster tiff file
                str
    bbox:       TODO!
    """
    
    path = Path(rasterpath)
    name = path.stem

    with rasterio.Env():
        with rasterio.open(rasterpath) as src:        
            meta = src.meta
                    
            native_bounds = transform_bounds(CRS.from_epsg(crs), 
                                             src.crs, *bbox)                
            bounds_window = src.window(*native_bounds)
            bounds_window = bounds_window.intersection(Window(0, 0, 
                                                              src.width, 
                                                              src.height))
        
            img = src.read(1, window=bounds_window)
            img[img == meta['nodata']] = 0
        
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(10,7))
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        show(img, ax=ax, cmap='gray', title=f"{name}")

    return img


