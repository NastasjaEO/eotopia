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

logging.basicConfig(level=logging.DEBUG)

testdatapath="D:/Code/eotopia/tests/testdata/TestEOPatch"

class TestEopatchCoords(unittest.TestCase):
    def test_newxarraycoords(self):
        eopatch = EOPatch.load(testdatapath)
        crs = eopatch.bbox.crs
        feature = eopatch.get_features()
        features = list(FeatureParser(feature))
        feature_type, feature_name = features[0]
        df = feature_array_to_dataframe(eopatch, 
                                             (feature_type, 
                                              feature_name), crs=crs)
        x_values, y_values = new_xarray_coordinates(df, crs, CRS.POP_WEB)
        self.assertEqual(len(x_values), 100)

if __name__ == '__main__':
    unittest.main()



