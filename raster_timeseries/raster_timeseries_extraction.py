# -*- coding: utf-8 -*-
"""
Created on Sat May 22 16:18:31 2021

@author: freeridingeo
"""

import rasterio 
from rasterio.windows import Window
import numpy as np
import pandas as pd
import geopandas as gpd

def extract_timeseries_at_pointcoordinate_to_df(bands, rasterpath, 
                                                xx, yy, time_index, 
                                                startindex=None):

    if startindex == None:
        num_bands = len(bands)
        num_ts = len(time_index)
        startindex = int(np.abs(num_ts-num_bands))
    else:
        startindex = 0

    zz = np.zeros(len(bands))
    for i in range(1, len(bands)):
        with rasterio.open(rasterpath) as src:
            ptx = xx
            pty = yy
            row, col = src.index(ptx, pty)
            val = src.read(i, window=Window(row, col, 1, 1))
            zz[i] = val
    df = pd.DataFrame(zz, index=time_index[startindex:])
    return df

def extract_timeseries_at_pointcoordinates_to_df(bands, rasterpath, 
                                                gdf, time_index, 
                                                startindex=None):

    if startindex == None:
        num_bands = len(bands)
        num_ts = len(time_index)
        startindex = int(np.abs(num_ts-num_bands))
    else:
        startindex = 0

    ds = rasterio.open(rasterpath)
    bbox = ds.bounds
    bands = ds.indexes

    sub = [] 
    for i in range(len(gdf)):
        point = gdf.geometry.iloc[i]
        px = point.x
        py = point.y
        if px >= bbox[0] and px <= bbox[2] and py >= bbox[1] and py <= bbox[3]:
            sub.append(gdf.iloc[i])
        sub_arr = np.asarray(sub)

        gdf_sub = gpd.GeoDataFrame(sub_arr)
        gdf_sub.columns = gdf.columns

    zz = np.zeros((len(gdf_sub), len(bands)))
    for i in range(len(gdf_sub)):
        for j in range(len(bands)):
            with rasterio.open(rasterpath) as src:
                point = gdf_sub.geometry.iloc[i]
                ptx = point.x
                pty = point.y
                row, col = src.index(ptx, pty)
                val = src.read(i, window=Window(row, col, 1, 1))
                if val.size != 0:
                    zz[i, j] = val
                else:
                    zz[i, j] = np.nan
    df = pd.DataFrame(zz, index=time_index[startindex:])
    return df

