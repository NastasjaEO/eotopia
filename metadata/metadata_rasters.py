# -*- coding: utf-8 -*-
"""
Created on Thu May 13 14:59:47 2021

@author: freeridingeo
"""

import pandas as pd
import rasterio
from rasterio.transform import Affine, array_bounds

RASTER_META_PARAMS = [
    "width",
    "height",
    "count",
    "crs",
    "transform",
    "dtype",
    "nodata",
    "driver",
]

RASTER_BOUND_PARAMS = ["left", "bottom", "right", "top"]
RASTER_GRID_PARAMS = ["crs", "width", "height"] + RASTER_BOUND_PARAMS
RASTER_RESOLUTION_PARAMS = ["xres", "yres"]
RASTER_FILE_PARAMS = ["file_path", "image_size", "file_size", "file_update"]
RASTER_GEOMETRY_PARAMS = ["bounding_box"]

RASTER_PROFILE_PARAMS = RASTER_META_PARAMS + [
    "tiled",
    "compress",
    "BIGTIFF",
    "interleave",
]

RASTER_FULL_PROFILE_PARAMS = (
    RASTER_PROFILE_PARAMS
    + RASTER_BOUND_PARAMS
    + RASTER_RESOLUTION_PARAMS
    + RASTER_FILE_PARAMS
    + RASTER_GEOMETRY_PARAMS
)

DEFAULT_GTIFF_PARAMS = dict(
    tiled=False,
    driver="GTiff",
    compress="deflate",
    interleave="band",
    BIGTIFF="IF_SAFER",
)

def rasterfile_properties(rasterpath):

    ds = rasterio.open(rasterpath)
    metadata = ds.profile
    properties = {}
    bands = ds.indexes
    properties["Number of bands"] = len(bands)
    raster_bounds = ds.bounds
    properties["Minimum longitude"] = raster_bounds[0]
    properties["Minimum latitude"] = raster_bounds[1]
    properties["Maximum longitude"] = raster_bounds[2]
    properties["Maximum latitude"] = raster_bounds[3]
    properties["Maximum latitude"] = raster_bounds[3]
    properties["Driver"] = metadata.data['driver']
    properties["Data type"] = metadata.data['dtype']
    properties["Width"] = metadata.data['width']
    properties["Height"] = metadata.data['height']
    properties["CRS"] = metadata.data['crs']
    properties["Interleave"] = metadata.data['pixel']
    return properties

def get_full_raster_profile(rasterpath, add_file_params=None, add_bounding_box=None):
    """
    rasterpath: Path
    Return profile data of raster image.
    """
    with rasterio.open(rasterpath) as src:
        raster_profile = dict(
            count=src.count,
            width=src.width,
            height=src.height,
            crs=src.crs["init"],
            xres=src.res[0],
            yres=src.res[1],
            left=src.bounds[0],
            right=src.bounds[2],
            top=src.bounds[3],
            bottom=src.bounds[1],
            transform=src.transform,
            dtype=src.meta["dtype"],
            nodata=src.nodata,
            driver=src.driver,
            compress=src.compression,
            tiled=src.profile["tiled"],
            interleave=src.profile["interleave"]
            if "interleave" in src.profile
            else None,
            BIGTIFF=src.profile["BIGTIFF"] if "BIGTIFF" in src.profile else None,
        )
    return pd.Series(raster_profile, name="profile")

def get_raster_bounds(raster_meta):
    """Return raster bound coordinates from raster meta."""
    left, bottom, right, top = array_bounds(
        height=raster_meta["height"],
        width=raster_meta["width"],
        transform=raster_meta["transform"],
    )
    return left, bottom, right, top

