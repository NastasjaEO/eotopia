# -*- coding: utf-8 -*-
"""
Created on Fri May 21 11:55:50 2021

@author: freeridingeo
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from sentinelhub import BBoxSplitter, CRS
import rasterio
from rasterio.windows import Window, bounds as wind_bounds
from rasterio.warp import transform_bounds
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
    
    if aoi_geo.crs == t_crs:
        aoi_reprj = aoi_geo.to_crs(crs=t_crs.pyproj_crs())
    else:
        aoi_reprj = aoi_geo
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

def create_eopatch_tiles_from_aoi_pixels(aoi_raster, t_crs, 
                                         res=10, grid_sz = 46):
    """
    Loop through aoi pixels in the geotif to create grid cells. 
    
    ---
    Param
    
    aoi_raster: geotif of aoi;
    t_crs: target CRS for the grid cell
    grid_sz: grid cell size.
    
    Return
    patchIDs: patch ID 
    tile_list: EOpatch that contain boundbox of grid cell
    
    """
    gpd_geo = list()
    prop = list()
    tile_lists = list()
    # loop through each row and column of aoi pixel to create bounding box
    with rasterio.open(aoi_raster) as src_dst:
        for col_off in range(0, src_dst.width):
            for row_off in range(0, src_dst.height):
                bounds = wind_bounds(Window(col_off, row_off, 1, 1), src_dst.transform)
                xmin, ymin, xmax, ymax = transform_bounds(
                    *[src_dst.crs, "epsg:4326"] + list(bounds), densify_pts=21
                )
                poly = Polygon([
                    (xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)
                ])
                gpd_geo.append(poly)
                prop.append("{}_{}".format(col_off, row_off))
    gpd_df = gpd.GeoDataFrame(prop, crs=CRS.WGS84.pyproj_crs(), geometry=gpd_geo)

    gpd_reproj = gpd_df.rename(columns={0: "id", "geometry": "geometry"})
    gpd_reproj = gpd_reproj.to_crs(crs=t_crs.pyproj_crs())
    designed_bbox_shapes = gpd_reproj.geometry.tolist()
    for aoi_shape in designed_bbox_shapes:
        width_pix = int((aoi_shape.bounds[2] - aoi_shape.bounds[0])/res)
        heigth_pix = int((aoi_shape.bounds[3] - aoi_shape.bounds[1])/res)

        width_grid = int(round(width_pix/grid_sz))
        heigth_grid =  int(round(heigth_pix/grid_sz))

        # split the tile grid by the desired grid number
        tile_splitter = BBoxSplitter([aoi_shape], t_crs, (width_grid, heigth_grid))

        tile_list = np.array(tile_splitter.get_bbox_list())
        info_list = np.array(tile_splitter.get_info_list())

        # get the all pylogon information from the splitted AOI
        idxs_x = [info['index_x'] for info in tile_splitter.info_list]
        idxs_y = [info['index_y'] for info in tile_splitter.info_list]
        tile_lists.append(tile_list)

    tile_list = np.array(tile_lists).flatten()
    return tile_list