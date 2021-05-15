# -*- coding: utf-8 -*-
"""
Created on Sat May 15 08:27:23 2021

@author: freeridingeo
"""

import unittest
import logging

from eolearn.core import EOPatch

import sys
sys.path.append("D:/Code/eotopia/utils")
from xarray_utils import eopatch_to_xrdataset

logging.basicConfig(level=logging.DEBUG)

testdatapath = "D:/Code/eotopia/tests/testdata/TestEOPatch"

class TestXarrayUtils(unittest.TestCase):
    def test_eopatch_to_xrdataset(self):
        eopatch = EOPatch.load(testdatapath)
        dataset = eopatch_to_xrdataset(eopatch)

if __name__ == '__main__':
    unittest.main()



