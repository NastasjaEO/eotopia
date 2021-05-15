# -*- coding: utf-8 -*-
"""
Created on Sat May 15 08:27:23 2021

@author: freeridingeo
"""

import unittest
import logging

from sentinelhub import CRS
from eolearn.core import EOPatch, FeatureParser

import sys
sys.path.append("D:/Code/eotopia/utils")
from coordinate_utils import new_xarray_coordinates
from dataframe_utils import feature_array_to_dataframe
sys.path.append("D:/Code/eotopia/visualization")
from eopatch_visualization import plot_eopatch_raster

logging.basicConfig(level=logging.DEBUG)

testdatapath="D:/Code/eotopia/tests/testdata/TestEOPatch"

class TestEopatchPlotting(unittest.TestCase):

    def test_plot_eopatch_raster(self):
        eopatch = EOPatch.load(testdatapath)
        
    def plot_eopatch_data(self):
        eopatch = EOPatch.load(testdatapath)


if __name__ == '__main__':
    unittest.main()



