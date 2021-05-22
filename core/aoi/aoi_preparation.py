# -*- coding: utf-8 -*-
"""
Created on Sat May 22 11:32:05 2021

@author: freeridingeo
"""

from pathlib import Path
import numpy as np
import geopandas as gpd
from sentinelhub import BBoxSplitter, CRS
from shapely.geometry import Polygon

def basic_aoi_preparation(vectorfile_path, 
                          crs="EPSG:4326", buffer_size=None, res=None,
                          gr_sz = None, split=False,
                          **kwargs):
    aoi = gpd.read_file(vectorfile_path)
    path = Path(vectorfile_path)
    outpath = path.home    
    
    inputs = []
    for arg in kwargs.values():
        inputs.append(arg)

    print(aoi.crs)
    if aoi.crs != crs:
        aoi = aoi.to_crs(crs)
    print(aoi.crs)
    
    if buffer_size != None:
        aoi = aoi.buffer(buffer_size)

    aoi_shape = aoi.geometry.values[-1]
    
    ShapeVal_a = round(aoi_shape.bounds[2] - aoi_shape.bounds[0])
    ShapeVal_b = round(aoi_shape.bounds[3] - aoi_shape.bounds[1])
    print(ShapeVal_a)
    print(ShapeVal_b)

    if split == True:
        SplitVal_a = max(1, int(ShapeVal_a/1e4))
        SplitVal_b = max(1, int(ShapeVal_b/1e4))
        print('The extent of the AOI is {}m x {}m,\
              so it is split into a grid of {} x {}.'.format(ShapeVal_a, 
                                                              ShapeVal_b,
                                                              SplitVal_a,
                                                              SplitVal_b))
        split_aoi_by_bbox(aoi, crs, ShapeVal_a, ShapeVal_b, outpath)


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

def split_aoi_by_bbox(aoi, crs, x_size, y_size, output_path):
    aoi_shape = aoi.geometry.values[-1]

    # split area of interest into an appropriate number of BBoxes
    bbox_splitter = BBoxSplitter([aoi_shape], crs, (x_size, y_size))
    bbox_list = np.array(bbox_splitter.get_bbox_list()) # get list of BBox geometries
    info_list = np.array(bbox_splitter.get_info_list()) # get list of x (column) and y(row) indices

    print(f'Each bounding box also has some info how it was created.\nExample:\n\
          bbox: {bbox_list[0].__repr__()} \n info: {info_list[0]}\n')

    geometry = [Polygon(bbox.get_polygon()) for bbox in bbox_list]
    idxs_x = [info['index_x'] for info in info_list] # get column index for naming EOPatch
    idxs_y = [info['index_y'] for info in info_list] # get row index for naming EOPatch

    gdf = gpd.GeoDataFrame({'index_x': idxs_x, 'index_y': idxs_y},
                       crs=crs.pyproj_crs(),
                       geometry=geometry)

    shapefile_name = output_path / 'BBoxes.geojson'
    gdf.to_file(str(shapefile_name), driver="GeoJSON")


path1 = "D:/Code/eotopia/tests/testdata/vectorfiles/small_test_aoi.shp"
path2 = "D:/Code/eotopia/tests/testdata/vectorfiles/large_test_aoi.shp"
path3 = "D:/Code/eotopia/tests/testdata/vectorfiles/eastern_france.geojson"
path4 = "D:/Code/eotopia/tests/testdata/vectorfiles/svn_border_3857.geojson"
path5 = "D:/Code/eotopia/tests/testdata/vectorfiles/svn_border_4326.geojson"

aoi = basic_aoi_preparation(path5 , crs = "EPSG:32632")

#aoi.plot()
