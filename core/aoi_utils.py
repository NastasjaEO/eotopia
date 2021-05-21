# -*- coding: utf-8 -*-
"""
Created on Fri May 21 11:55:50 2021

@author: freeridingeo
"""

import os

import numpy as np
import pandas as pd
import geopandas as gpd
from sentinelhub import BBoxSplitter
from shapely.geometry import Polygon

def prepare_large_aoi_grid(vectorfile, t_crs, gr_sz, res=10, save=False):
    """
    read in shape file and reproject it into the projection that will compute 
    correct aoi size
    
    Args:
        vectorfile: the AOI shapfile either in ESRI shapefile or geojson
        t_crs: the target coordination;
        gr_sz: tile/grid size to split the AOI, default is 168 by 168 pixels;
        save: save the generated AOI tile grid as a pickle file
    Return:
        patchID: the splitted tile that will be saved as EOpatch with the IDs

    Note:
        when save is set to Ture. An ESRI shapefile is saved to the disk under 
        folder called "aoi_tile_grid"
    """

    aoi_geo = gpd.read_file(vectorfile)
    aoi_reprj = aoi_geo.to_crs(crs=t_crs.pyproj_crs())
    aoi_shape = aoi_reprj.geometry.values[-1]

    data_res = res
    width_pix = int((aoi_shape.bounds[2] - aoi_shape.bounds[0])/data_res)
    heigth_pix = int((aoi_shape.bounds[3] - aoi_shape.bounds[1])/data_res)
    print('Dimension of the area is {} x {} pixels'\
          .format(width_pix, heigth_pix))
    width_grid = int(round(width_pix/gr_sz))
    heigth_grid = int(round(heigth_pix/gr_sz))

    tile_splitter = BBoxSplitter([aoi_shape], t_crs, (width_grid, heigth_grid))
    print("The area is splitted into a grid with {} by {} tiles!"\
          .format(width_grid, heigth_grid))

    tiles = np.array(tile_splitter.get_bbox_list())
    info_list = np.array(tile_splitter.get_info_list())

    # get the all polygon information from the splitted AOI
    idxs_x = [info['index_x'] for info in tile_splitter.info_list]
    idxs_y = [info['index_y'] for info in tile_splitter.info_list]

    #save all the patch ID for tiles and save it as numpy array
    patchID = np.array(range(len(tiles))).astype("int")
    geometry = [Polygon(bbox_.get_polygon()) for bbox_ in tiles[patchID]]

    while save == True:
        # get the name of the file
        nm = vectorfile.split("/")[-1]
        tile_path = "aoi_tile_grid"
        df = pd.DataFrame({'index_x': idxs_x, 'index_y': idxs_y})
        gdf = gpd.GeoDataFrame(df, crs=t_crs.pyproj_crs(), geometry= geometry)
        gdf.to_file(os.path.join(tile_path, nm))

    return patchID, tiles
