# -*- coding: utf-8 -*-
"""
Created on Wed May 12 14:26:49 2021

@author: freeridingeo
"""

import collections
import numpy as np
from eolearn.core import EOPatch, EOTask, FeatureType, FeatureTypeSet

RasterType = collections.namedtuple('RasterType', 
                                    'id unit sample_type np_dtype feature_type')


PREDEFINED_S2_TYPES = {
    RasterType("bool_mask", 'DN', 'UINT8', bool, FeatureType.MASK): [
            "dataMask"
    ],
    RasterType("mask", 'DN', 'UINT8', np.uint8, FeatureType.MASK): [
            "CLM", "SCL"
    ],
    RasterType("uint8_data", 'DN', 'UINT8', np.uint8, FeatureType.DATA): [
            "SNW", "CLD", "CLP"
    ],
    RasterType("bands", 'DN', 'UINT16', np.uint16, FeatureType.DATA): [
            "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12", "B13"
    ],
    RasterType("other", 'REFLECTANCE', 'FLOAT32', np.float32, FeatureType.DATA): [
            "sunAzimuthAngles", "sunZenithAngles", "viewAzimuthMean", "viewZenithMean"
    ]
}


