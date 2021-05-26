# -*- coding: utf-8 -*-
"""
Created on Sat May 22 16:03:56 2021

@author: freeridingeo
"""

from pathlib import Path
import numpy as np
import geopandas as gpd
from sentinelhub import BBoxSplitter
from shapely.geometry import Polygon

import matplotlib.pyplot as plt

def split_vector_by_bbox(vector, crs, x_size, y_size, 
                         reduce_bbox_sizes=True, output_path=None):
    vector_shape = vector.geometry.values[-1]

    # split area of interest into an appropriate number of BBoxes
    bbox_splitter = BBoxSplitter([vector_shape], crs, (x_size, y_size), reduce_bbox_sizes)
    bbox_list = np.array(bbox_splitter.get_bbox_list()) # get list of BBox geometries
    info_list = np.array(bbox_splitter.get_info_list()) # get list of x (column) and y(row) indices
    geometry_list = bbox_splitter.get_geometry_list()

    print(f'Each bounding box also has some info how it was created.\nExample:\n\
          bbox: {bbox_list[0].__repr__()} \n info: {info_list[0]}\n')

    geometry = [Polygon(bbox.get_polygon()) for bbox in bbox_list]
    idxs_x = [info['index_x'] for info in info_list] # get column index for naming EOPatch
    idxs_y = [info['index_y'] for info in info_list] # get row index for naming EOPatch

    gdf = gpd.GeoDataFrame({'index_x': idxs_x, 'index_y': idxs_y},
                       crs=crs,
                       geometry=geometry)
    fig, ax = plt.subplots(figsize=(20, 20))
    gdf.plot(ax=ax, facecolor='w', edgecolor='r', alpha=0.5, linewidth=5)
    vector.plot(ax=ax, facecolor='w', edgecolor='k', alpha=0.5)
    ax.set_title('Vectorfile Splitted');
    plt.axis('off')
    plt.xticks([]);
    plt.yticks([]);

    if output_path:
        shapefile_name = output_path / 'BBoxes.geojson'
        gdf.to_file(str(shapefile_name), driver="GeoJSON")
    return gdf, geometry_list

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
    fig, ax = plt.subplots(figsize=(20, 20))
    gdf.plot(ax=ax, facecolor='w', edgecolor='r', alpha=0.5, linewidth=5)
    vector.plot(ax=ax, facecolor='w', edgecolor='k', alpha=0.5)
    ax.set_title('Vectorfile Splitted');
    plt.axis('off')
    plt.xticks([]);
    plt.yticks([]);

    patchIDs = check_patch_size(bbox_list, info_list, ID, size)
    # Change the order of the patches (useful for plotting)
    patchIDs = np.transpose(np.fliplr(np.array(patchIDs)\
                                      .reshape(int(size/1e4), int(size/1e4))))\
                                    .ravel()

    if output_path:
        shapefile_name = output_path / 'BBoxes.geojson'
        gdf.to_file(str(shapefile_name), driver="GeoJSON")
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

def show_splitter(splitter, alpha=0.2, area_buffer=0.2, show_legend=False):
    area_bbox = splitter.get_area_bbox()
    minx, miny, maxx, maxy = area_bbox
    lng, lat = area_bbox.middle
    w, h = maxx - minx, maxy - miny
    minx = minx - area_buffer * w
    miny = miny - area_buffer * h
    maxx = maxx + area_buffer * w
    maxy = maxy + area_buffer * h

    fig=plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
