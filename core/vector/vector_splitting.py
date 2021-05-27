# -*- coding: utf-8 -*-
"""
Created on Sat May 22 16:03:56 2021

@author: freeridingeo
"""

from pathlib import Path
import numpy as np
import geopandas as gpd
from sentinelhub import (BBoxSplitter, OsmSplitter, TileSplitter, 
                         UtmGridSplitter, UtmZoneSplitter)
from shapely.geometry import Polygon

import matplotlib.pyplot as plt


def split_vector_by_area(vector, crs, dict_of_criteria, 
                         how="bbox",
                         reduce_bbox=True, outpath=None):
    """
    vector:         GeoDataFrame
    
    Returns:
        gdf         GeoDataFrame with split infos
        splitter    Splitter object which contains more info
    """
    vector_shape = vector.geometry.values[-1]
    if how == 'bbox':
        x_size =  dict_of_criteria['x_size']
        y_size =  dict_of_criteria['y_size']
        splitter = BBoxSplitter([vector_shape], crs, 
                                (x_size, y_size), 
                                reduce_bbox_sizes = reduce_bbox)

    elif how == 'osm':
        zoom = dict_of_criteria['zoom']
        splitter = OsmSplitter([vector_shape], crs, zoom_level = zoom, 
                                reduce_bbox_sizes = reduce_bbox)

    gdf = get_splitter_gdf(vector, splitter, crs, outpath, plot=True)
    return gdf, splitter

def split_vector_by_utm(vector, crs, dict_of_criteria, 
                         how="zone",
                         reduce_bbox=True, outpath=None):
    """
    vector:         GeoDataFrame
    
    Returns:
        gdf         GeoDataFrame with split infos
        splitter    Splitter object which contains more info
    """
    vector_shape = vector.geometry.values[-1]
    if how == 'zone':
        x_size =  dict_of_criteria['x_size']
        y_size =  dict_of_criteria['y_size']
        splitter = UtmZoneSplitter([vector_shape], crs, 
                                (x_size, y_size), 
                                reduce_bbox_sizes = reduce_bbox)
    elif how == 'grid':
        x_size =  dict_of_criteria['x_size']
        y_size =  dict_of_criteria['y_size']
        splitter = UtmGridSplitter([vector_shape], crs, 
                                (x_size, y_size), 
                                reduce_bbox_sizes = reduce_bbox)

    gdf = get_splitter_gdf(vector, splitter, crs, outpath, plot=True)
    return gdf, splitter

def split_vector_by_tile(vector, crs, time_interval, tile_split_shape=1, 
                         data_collection=None, config=None,
                         reduce_bbox=True, outpath=None):
    """
    vector:         GeoDataFrame

    time_interval:  Interval with start and end date of the form 
                    YYYY-MM-DDThh:mm:ss or YYYY-MM-DD
                    (str, str)
    tile_split_shape: Parameter that describes the shape in which the satellite 
            tile bounding boxes will be split. It can be a tuple of the form 
            `(n, m)` which means the tile bounding boxes will be split into `n` 
            columns and `m` rows. It can also be a single integer `n` which is the same
            as `(n, n)`.
            int or (int, int)
    data_collection: A satellite data collection (sentinelhub)
            DataCollection
    config: A custom instance of config class to override parameters from the 
        saved configuration.
            SHConfig or None
    reduce_bbox_sizes: If `True` it will reduce the sizes of bounding boxes so 
        that they will tightly fit the given area geometry from `shape_list`.
    
    Returns:
        gdf         GeoDataFrame with split infos
        splitter    Splitter object which contains more info
    """
    vector_shape = vector.geometry.values[-1]
    splitter = TileSplitter([vector_shape], crs, time_interval, tile_split_shape,
                            data_collection, config,
                            reduce_bbox_sizes = reduce_bbox)
    
    gdf = get_splitter_gdf(vector, splitter, crs, outpath, plot=True)
    return gdf, splitter

def get_splitter_gdf(vector, splitter, crs, outpath=None, plot=False):
    bbox_list = np.array(splitter.get_bbox_list())
    info_list = np.array(splitter.get_info_list())
    idxs_x = [info['index_x'] for info in info_list] # get column index
    idxs_y = [info['index_y'] for info in info_list] # get row index

    geometry = [Polygon(bbox.get_polygon()) for bbox in bbox_list]

    gdf = gpd.GeoDataFrame({'index_x': idxs_x, 'index_y': idxs_y},
                       crs=crs,
                       geometry=geometry)

    if plot == True:
        fig, ax = plt.subplots(figsize=(20, 20))
        gdf.plot(ax=ax, facecolor='w', edgecolor='r', alpha=0.5, linewidth=5)
        vector.plot(ax=ax, facecolor='w', edgecolor='k', alpha=0.5)
        ax.set_title('Vectorfile Splitted');
        plt.axis('off')
        plt.xticks([]);
        plt.yticks([]);

    if outpath:
        shapefile_name = outpath / 'BBoxes.geojson'
        gdf.to_file(str(shapefile_name), driver="GeoJSON")
    return gdf

def get_splitter_info(splitter):
    info_list = splitter.get_info_list()
    bbox_list = splitter.get_bbox_list()
    geometry_list = splitter.get_geometry_list()
    area_shape = splitter.get_area_shape()
    area_bbox = splitter.get_area_bbox()
    return area_bbox, area_shape, bbox_list, geometry_list, info_list

def split_vector_by_bbox_with_specified_size(vector, size, ID, 
                                          reduce_bbox_sizes=True, output_path=None):
    """ 
    size:   int
            bbox side length in meters
    """
    from sentinelhub import UtmZoneSplitter
    vector_shape = vector.geometry.values[-1]
    bbox_splitter = UtmZoneSplitter([vector_shape], vector.crs, size, 
                                    reduce_bbox_sizes)

    bbox_list = np.array(bbox_splitter.get_bbox_list()) # get list of BBox geometries
    info_list = np.array(bbox_splitter.get_info_list()) # get list of x (column) and y(row) indices
    geometry_list = bbox_splitter.get_geometry_list()

    print(f'Each bounding box also has some info how it was created.\nExample:\n\
          bbox: {bbox_list[0].__repr__()} \n info: {info_list[0]}\n')

    geometry = [Polygon(bbox.get_polygon()) for bbox in bbox_list]
    idxs_x = [info['index_x'] for info in info_list] # get column index for naming EOPatch
    idxs_y = [info['index_y'] for info in info_list] # get row index for naming EOPatch

    gdf = gpd.GeoDataFrame({'index_x': idxs_x, 'index_y': idxs_y},
                       crs=vector.crs,
                       geometry=geometry)

    patchIDs = check_patch_size(bbox_list, info_list, ID, size)
    # Change the order of the patches (useful for plotting)
    patchIDs = np.transpose(np.fliplr(np.array(patchIDs)\
                                      .reshape(int(size/1e4), int(size/1e4))))\
                                    .ravel()

    return gdf, geometry_list

def check_patch_size(bbox_list, info_list, ID, size):
    _size = int(size/1e4)
    patchIDs = []
    for idx, (bbox, info) in enumerate(zip(bbox_list, info_list)):
        if (abs(info['index_x'] - info_list[ID]['index_x']) <= 2 and
            abs(info['index_y'] - info_list[ID]['index_y']) <= 2):
            patchIDs.append(idx)
    if len(patchIDs) != _size*_size:
        print('Warning! Use a different central patch ID,' 
              'this one is on the border.')
    return patchIDs

