# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 11:44:16 2021

@author: freeridingeo
"""

import numpy as np
import pandas as pd
import rasterio

def extract_rasterdata_to_csv(inputpath, outputpath, timeindex=None):
    """
    

    Parameters
    ----------
    inputpath : TYPE
        DESCRIPTION.
    outputpath : TYPE
        DESCRIPTION.
    timeindex: TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """

    ds = rasterio.open(inputpath)
    num_bands = ds.count
    width = ds.width
    height = ds.height

    values_2D = []
    for i in range(1,num_bands):
        with rasterio.open(inputpath) as src:
            values = src.read(i)
            values_2D.append(values)

    latitudes = np.zeros((width, height))
    longitudes = np.zeros((width, height))
    with rasterio.open(inputpath) as src:
        for i in range(width):
            for j in range(height):
                long, lat = src.xy(j, i)
                latitudes[i, j] = lat
                longitudes[i, j] = long

    lats = latitudes.flatten()
    longs = longitudes.flatten()
    arrays = [
            np.round(lats,5),
            np.round(longs,5),
            ]
    tuples = list(zip(*arrays))
    index = pd.MultiIndex.from_tuples(tuples, names=["Latitude", "Longitude"])

    values = []
    for i in range(len(values_2D)):
        val = values_2D[i].flatten()
        values .append(val)

    if timeindex:
        timeindex = pd.to_datetime(timeindex, format = '%Y%m%d', errors = 'ignore')
        df = pd.DataFrame(values, index=timeindex[1:] , columns=index)
    else:
        df = pd.DataFrame(values, columns=index)

    df.to_csv(outputpath)

