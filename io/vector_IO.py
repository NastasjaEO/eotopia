# -*- coding: utf-8 -*-
"""
Created on Sat May 22 15:21:20 2021

@author: freeridingeo
"""

from pathlib import Path
from sentinelhub import BBox, CRS

def read_wkt_file(vector_path):
    import shapely.wkt

    with open(vector_path, 'r') as f:
        shape = f.read()
    geometry = shapely.wkt.loads(shape)
    return shape, geometry

def wkt_to_bbox(vector_path, inflate_bbox=None):
    """ 
    inflate: float
            e.g. 0.1
    """
    shape, geometry = read_wkt_file(vector_path)
    minx, miny, maxx, maxy = geometry.bounds
    
    if not inflate_bbox:
        inflate_bbox = 0.0

    delx = maxx - minx
    dely = maxy - miny
    minx = minx - delx * inflate_bbox
    maxx = maxx + delx * inflate_bbox
    miny = miny - dely * inflate_bbox
    maxy = maxy + dely * inflate_bbox
    
    bbox = BBox([minx, miny, maxx, maxy], crs=CRS.WGS84)
    return bbox


path1 = "D:/Code/eotopia/tests/testdata/vectorfiles/theewaterskloof_dam_nominal.wkt"
shape, geometry = read_wkt_file(path1)
