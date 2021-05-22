# -*- coding: utf-8 -*-
"""
Created on Sat May 22 11:32:05 2021

@author: freeridingeo
"""

import geopandas as gpd

def basic_aoi_preparation(vectorfile_path, 
                          crs="EPSG:4326", buffer_size=.001, 
                          **kwargs):
    aoi = gpd.read_file(vectorfile_path)
    
    inputs = []
    for arg in kwargs.values():
        inputs.append(arg)
    
    if aoi.crs != crs:
        aoi = aoi.to_crs(crs)
    
    if "buffer "in inputs:
        aoi = aoi.buffer(buffer_size)
