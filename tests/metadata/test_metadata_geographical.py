# -*- coding: utf-8 -*-
"""
Created on Sat May 15 08:27:23 2021

@author: freeridingeo
"""

import unittest
import logging

from eolearn.core import EOPatch, FeatureType

import sys
sys.path.append("D:/Code/eotopia/metadata")
from metadata_geographical import (get_eopatch_bbox,
                                   _get_spatial_coordinates_of_eopatch,
                                   get_eopatch_coordinates)

logging.basicConfig(level=logging.DEBUG)

testdatapath = "D:/Code/eotopia/tests/testdata/TestEOPatch"

class TestEOPatchGeographicalMeta(unittest.TestCase):

    def test_get_eopatch_bbox(self):
        eopatch = EOPatch.load(testdatapath)
        bbox = get_eopatch_bbox(eopatch)

    def test_get_spatial_coordinates_of_eopatch(self):
        eopatch = EOPatch.load(testdatapath)
        bbox = get_eopatch_bbox(eopatch)
        featuretype = FeatureType.DATA_TIMELESS
        data = eopatch[FeatureType.DATA_TIMELESS]['DEM']
        coords = _get_spatial_coordinates_of_eopatch(bbox, data, featuretype)
        self.assertEqual(len(coords['x']), 100)

    def test_get_eopatch_coordinates(self):
        eopatch = EOPatch.load(testdatapath)
        crs = eopatch.bbox.crs
        features = eopatch.get_feature_list()
        for feature in features:
            if not isinstance(feature, tuple):
                continue
            get_eopatch_coordinates(eopatch, feature, crs)

if __name__ == '__main__':
    unittest.main()



