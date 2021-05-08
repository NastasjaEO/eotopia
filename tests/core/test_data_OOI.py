# -*- coding: utf-8 -*-
"""
Created on Thu May  6 19:38:36 2021

@author: nasta
"""

import os
import unittest
import logging
import warnings

import numpy as np
import geopandas as gpd

import dateutil.parser
import datetime

from sentinelhub import BBox, CRS

## TODO!
#from eotopia.core import data_types, data_OOI_IO

import sys
sys.path.append("D:/Code/eotopia/core")
import data_OOI
import data_OOI_IO
import data_types

logging.basicConfig(level=logging.DEBUG)

class TestEOPatchFeatureTypes(unittest.TestCase):

    PATCH_FILENAME = 'D:/Code/eotopia/tests/testdata/TestOOI'

    def test_loading_valid(self):
        eop = data_OOI.OOI.load(self.PATCH_FILENAME)



if __name__ == '__main__':
    unittest.main()

