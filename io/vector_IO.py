# -*- coding: utf-8 -*-
"""
Created on Sat May 22 15:21:20 2021

@author: freeridingeo
"""

from pathlib import Path

def read_wkt_file(vector_path):
    import shapely.wkt

    with open(vector_path, 'r') as f:
        shape = f.read()
    geometry = shapely.wkt.loads(shape)
    return shape, geometry

path1 = "D:/Code/eotopia/tests/testdata/vectorfiles/theewaterskloof_dam_nominal.wkt"

shape, geometry = read_wkt_file(path1)
