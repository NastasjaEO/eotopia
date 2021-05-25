# -*- coding: utf-8 -*-
"""
Created on Tue May 25 11:50:41 2021

@author: freeridingeo
"""

import unittest
import logging

import sys
sys.path.append("D:/Code/eotopia/core")
from data_classes import (RasterData, VectorData)

logging.basicConfig(level=logging.DEBUG)

rasterdatapath="D:/Code/eotopia/tests/testdata/rasterdata/raster1.tiff"
vectordatapath="D:/Code/eotopia/tests/testdata/vectorfiles"

class TestRasterData(unittest.TestCase):
    scene = RasterData(rasterdatapath)
    def testrasterproperties(self):        
        scene = RasterData(rasterdatapath)
        print("init", scene, "\n")
        print("area", scene.area, "\n")
        print("bandnames", scene.bandnames, "\n")
        print("bounds", scene.bbox, "\n")
        print("extent", scene.extent, "\n")
        print("geo", scene.geo, "\n")
    
if __name__ == '__main__':
    unittest.main()

