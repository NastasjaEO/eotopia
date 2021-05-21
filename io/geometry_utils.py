# -*- coding: utf-8 -*-
"""
Created on Fri May 21 11:07:09 2021

@author: freeridingeo
"""

from sentinelhub import BBox, bbox_to_dimensions, CRS

def raster_size_from_aoi_bbox(list_of_coords, resolution):
    """
    Example for list_of_coords: [13.353882, 45.402307, 16.644287, 46.908998]
    """
    aoi_bbox = BBox(list_of_coords, crs=CRS.WGS84)
    aoi_size = bbox_to_dimensions(aoi_bbox, resolution)
    return aoi_size


