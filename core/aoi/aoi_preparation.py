# -*- coding: utf-8 -*-
"""
Created on Sat May 22 11:32:05 2021

@author: freeridingeo
"""

import sys
from pathlib import Path
import geopandas as gpd
from sentinelhub import BBoxSplitter, CRS

sys.path.append("D:/Code/eotopia/core/vector")
from aoi_splitting import (split_vector_by_bbox, 
                           split_vector_by_bbox_with_specified_size)

def basic_aoi_preparation(vectorfile_path, 
                          crs="EPSG:4326", buffer_size=None, res=None,
                          gr_sz = None, split=False, ID=None,
                          **kwargs):
    aoi = gpd.read_file(vectorfile_path)
    path = Path(vectorfile_path)
    outpath = path.home    
    
    inputs = []
    for arg in kwargs.values():
        inputs.append(arg)

    print("AOI crs then",  aoi.crs)
    if aoi.crs != crs:
        aoi = aoi.to_crs(crs)
    print("AOI crs now",  aoi.crs, "\n")
    print("AOI bounds",  aoi.geometry[0].bounds, "\n")

    if buffer_size != None:
        aoi = aoi.buffer(buffer_size)

    aoi_shape = aoi.geometry.values[-1]
        
    ShapeVal_a = round(aoi_shape.bounds[2] - aoi_shape.bounds[0])
    ShapeVal_b = round(aoi_shape.bounds[3] - aoi_shape.bounds[1])


    if split == True:
        SplitVal_a = max(1, int(ShapeVal_a/1e4))
        SplitVal_b = max(1, int(ShapeVal_b/1e4))
        print('The extent of the AOI is {}m x {}m,\
              so it is split into a grid of {} x {}.'.format(ShapeVal_a, 
                                                              ShapeVal_b,
                                                              SplitVal_a,
                                                              SplitVal_b))
        gdf = split_vector_by_bbox(aoi, crs, ShapeVal_a, ShapeVal_b, outpath)

    if isinstance(split, int):
        gdf = split_vector_by_bbox_with_specified_size(aoi, split, ID)

    if res != None:
        width_pix = int((aoi_shape.bounds[2] - aoi_shape.bounds[0])/res)
        height_pix = int((aoi_shape.bounds[3] - aoi_shape.bounds[1])/res)
        print('Dimension of the area is {} x {} pixels'\
              .format(width_pix, height_pix))
        
        if gr_sz != None:
            width_grid = int(round(width_pix/gr_sz))
            heigth_grid = int(round(height_pix/gr_sz))
            print('Dimension of the grid is {} x {}'\
              .format(width_grid, heigth_grid))

    return aoi


path1 = "D:/Code/eotopia/tests/testdata/vectorfiles/small_test_aoi.shp"
path2 = "D:/Code/eotopia/tests/testdata/vectorfiles/large_test_aoi.shp"
# path3 = "D:/Code/eotopia/tests/testdata/vectorfiles/eastern_france.geojson"
path4 = "D:/Code/eotopia/tests/testdata/vectorfiles/svn_border_3857.geojson"
path5 = "D:/Code/eotopia/tests/testdata/vectorfiles/svn_border_4326.geojson"

# aoi = basic_aoi_preparation(path1 , crs = "EPSG:32632", split=False)

testaoi = gpd.read_file(path5)
crs = "EPSG:32633"
testaoi = testaoi.to_crs(crs)

size = 5000
