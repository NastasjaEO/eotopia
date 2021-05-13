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

def read_rasterfile(path):
    dataset = rasterio.open(path)

    name = dataset.name
    print("Name of dataset:", name)    
    bands = dataset.count
    print("Number of bands:", bands)

    width = dataset.width
    height = dataset.height
    bounds = dataset.bounds
    print("The dataset has the following width and height", width, height, "\n")
    print("The dataset has the following bounds ", bounds)

    {i: dtype for i, dtype in zip(dataset.indexes, dataset.dtypes)}    
    crs = dataset.crs
    print("The dataset has the following reference system ", crs)    
    if bands <= 1:
        result = dataset.read(bands)
    else:
        result = []
        for i in range(1, bands):
            result.append(dataset.read(i))
    dataset = None
    return result

def read_rasterfile_xarray(rasterpath):
    import xarray as xr
    dataset = xr.open_dataset(rasterpath)
    return dataset

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

def sliced_raster_read(rasterpath, xmin, xmax, ymin, ymax, band=None):
    slice_ = (slice(xmin,xmax), slice(ymin,ymax))
    window_slice = Window.from_slices(*slice_)

    if band:
        from matplotlib.patches import Rectangle

        src = rasterio.open(rasterpath)    
        ds = src.read(band)
        plt.imshow(ds)
        ax = plt.gca()
        ax.add_patch(Rectangle((window_slice.col_off,window_slice.row_off),
                       width=window_slice.width,
                       height=window_slice.height,fill=True,alpha=.2,
                 color="red"))

    return window_slice

def read_rasterfile_xarray_by_bbox(rasterpath, bbox):
    import xarray as xr
    dataset = xr.open_dataset(rasterpath)
    if isinstance(bbox, int):
        dataset = dataset[dict(lat=slice(bbox[0], bbox[1]), 
                                   lon=slice(bbox[2], bbox[3]))]
    elif isinstance(bbox, float):
        dataset = dataset.loc[dict(lat=slice(bbox[0], bbox[1]), 
                                   lon=slice(bbox[2], bbox[3]))]
    return dataset

def get_raster_meta(rasterpath):
    """
    rasterpath: Path
    Return meta data of raster image."""
    with rasterio.open(rasterpath) as src:
        return src.meta

def get_raster_profile(rasterpath):
    """
    rasterpath: Path
    Return profile data of raster image."""
    with rasterio.open(rasterpath) as src:
        return src.profile

def write_numpydata_to_tiff(data, path, meta=None):
    """Write numpy data to tiff file."""
    
    if meta:
        with rasterio.open(path, 'w', **meta) as dst:
            dst.write(data)
    else:
        with rasterio.open(path, 'w', driver='GTiff', dtype=data.dtype) as dst:
            dst.write(data)


