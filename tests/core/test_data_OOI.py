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

class TestOOIDataTypes(unittest.TestCase):

    PATCH_FILENAME = 'D:/Code/eotopia/tests/testdata/TestOOI'

    def test_loading_valid(self):
        eop = data_OOI.OOI.load(self.PATCH_FILENAME)
        repr_str = eop.__repr__()
        self.assertTrue(isinstance(repr_str, str) and len(repr_str) > 0,
                        msg='OOI __repr__ must return non-empty string')

    def test_numpy_data_types(self):
        eop = data_OOI.OOI()

        data_examples = []
        for size in range(6):
            for dtype in [np.float32, np.float64, float, np.uint8, np.int64, bool]:
                data_examples.append(np.zeros((2, ) * size, dtype=dtype))

        for data_type in data_types.DataTypeSet.RASTER_TYPES:
            valid_count = 0

            for data in data_examples:
                try:
                    eop[data_type]['TEST'] = data
                    valid_count += 1
                except ValueError:
                    pass

            self.assertEqual(valid_count, 6,  # 3 * (2 - feature_type.is_discrete()),
                             msg='Data type {} should take only a specific\
                                 type of data'.format(data_type))

    def test_vector_data_types(self):
        eop = data_OOI.OOI()

        invalid_entries = [
            {}, [], 0, None
        ]

        for data_type in data_types.DataTypeSet.VECTOR_TYPES:
            for entry in invalid_entries:
                with self.assertRaises(ValueError,
                                       msg='Invalid entry {} for {} should\
                                           raise an error'.\
                                               format(entry, data_type)):
                    eop[data_type]['TEST'] = entry

        crs_test = CRS.WGS84.pyproj_crs()
        geo_test = gpd.GeoSeries([BBox((1, 2, 3, 4), 
                                       crs=CRS.WGS84).geometry], crs=crs_test)

        eop.vector_timeless['TEST'] = geo_test
        self.assertTrue(isinstance(eop.vector_timeless['TEST'], gpd.GeoDataFrame),
                        'GeoSeries should be parsed into GeoDataFrame')
        self.assertTrue(hasattr(eop.vector_timeless['TEST'], 'geometry'), 
                        'Feature should have geometry attribute')
        self.assertEqual(eop.vector_timeless['TEST'].crs, crs_test, 
                         'GeoDataFrame should still contain the crs')

        with self.assertRaises(ValueError, msg='Should fail because there\
                               is no TIMESTAMP column'):
            eop.vector['TEST'] = geo_test

    def test_bbox_data_type(self):
        eop = data_OOI.OOI()
        invalid_entries = [
            0, list(range(4)), tuple(range(5)), {}, set(), [1, 2, 4, 3, 4326, 3],\
                'BBox'
        ]
        for entry in invalid_entries:
            with self.assertRaises((ValueError, TypeError),
                                  msg='Invalid bbox entry {} should raise\
                                        an error'.format(entry)):
                eop.bbox = entry

    def test_timestamp_data_type(self):
        eop = data_OOI.OOI()
        invalid_entries = [
            [datetime.datetime(2017, 1, 1, 10, 4, 7), None, 
                 datetime.datetime(2017, 1, 11, 10, 3, 51)],
            'something',
            datetime.datetime(2017, 1, 1, 10, 4, 7)
        ]

        valid_entries = [
            ['2018-01-01', '15.2.1992'],
            (datetime.datetime(2017, 1, 1, 10, 4, 7), 
                 datetime.date(2017, 1, 11))
        ]

        for entry in invalid_entries:
            with self.assertRaises((ValueError, TypeError),
                                   msg='Invalid timestamp entry {} should\
                                       raise an error'.format(entry)):
                eop.timestamp = entry

        for entry in valid_entries:
            eop.timestamp = entry

    def test_invalid_characters(self):
        eopatch = data_OOI.OOI()

        with self.assertRaises(ValueError):
            eopatch.data_timeless['mask.npy'] = np.arange(3 * 3 * 2).reshape(3, 3, 2)

    ## TODO!
    def test_repr_no_crs(self):
        eop = data_OOI.OOI.load(self.PATCH_FILENAME)
        eop.vector_timeless["LULC"].crs = None
        repr_str = eop.__repr__()
        self.assertTrue(isinstance(repr_str, str) and len(repr_str) > 0,
                        msg='EOPatch __repr__ must return non-empty string\
                            even in case of missing crs')

class TestOOI(unittest.TestCase):

    PATCH_FILENAME = 'D:/Code/eotopia/tests/testdata/TestOOI'

    def test_add_data(self):
        bands = np.arange(2*3*3*2).reshape(2, 3, 3, 2)
        eop = data_OOI.OOI()
        eop.data['bands'] = bands

        self.assertTrue(np.array_equal(eop.data['bands'], bands), 
                        msg="Data numpy array not stored")

    def test_simplified_data_operations(self):
        bands = np.arange(2 * 3 * 3 * 2).reshape(2, 3, 3, 2)
        data = data_types.DataType.DATA, 'TEST-BANDS'
        eop = data_OOI.OOI()

        eop[data] = bands
        self.assertTrue(np.array_equal(eop[data], bands), 
                        msg="Data numpy array not stored")

    ## TODO!
    def test_rename_data(self):
         bands = np.arange(2 * 3 * 3 * 2).reshape(2, 3, 3, 2)
         eop = data_OOI.OOI()
         eop.data['bands'] = bands
    #     eop.rename_feature(data_types.DataType.DATA, 'bands', 'new_bands')
    #     self.assertTrue('new_bands' in eop.data)

    ## TODO!
    def test_rename_data_missing(self):
        bands = np.arange(2 * 3 * 3 * 2).reshape(2, 3, 3, 2)

        eop = data_OOI.OOI()
        eop.data['bands'] = bands
#        with self.assertRaises(BaseException,
#                               msg='Should fail because there is no `missing_bands` feature in the EOPatch.'):
        #     eop.rename_feature(data_types.DataType.DATA, 'missing_bands', 'new_bands')

    def test_get_feature(self):
        bands = np.arange(2*3*3*2).reshape(2, 3, 3, 2)

        eop = data_OOI.OOI()
        eop.data['bands'] = bands
        eop_bands = eop.get_feature(data_types.DataType.DATA, 'bands')
        self.assertTrue(np.array_equal(eop_bands, bands), 
                        msg="Data numpy array not returned properly")

    def test_remove_data(self):
        bands = np.arange(2*3*3*2).reshape(2, 3, 3, 2)
        names = ['bands1', 'bands2', 'bands3']

        eop = data_OOI.OOI()
        eop.add_feature(data_types.DataType.DATA, names[0], bands)
        eop.data[names[1]] = bands
        eop[data_types.DataType.DATA][names[2]] = bands

        for ooi_name in names:
            self.assertTrue(ooi_name in eop.data, "Data {} was not added to OOI".\
                            format(ooi_name))
            self.assertTrue(np.array_equal(eop.data[ooi_name], bands), 
                            "Data of data {} is incorrect".format(ooi_name))

        eop.remove_feature(data_types.DataType.DATA, names[0])
        del eop.data[names[1]]
        del eop[data_types.DataType.DATA][names[2]]
        for ooi_name in names:
            self.assertFalse(ooi_name in eop.data, 
                             msg="Data {} should be deleted from "
                             "OOI".format(ooi_name))

    def test_concatenate(self):
        eop1 = data_OOI.OOI()
        bands1 = np.arange(2*3*3*2).reshape(2, 3, 3, 2)
        eop1.data['bands'] = bands1

        eop2 = data_OOI.OOI()
        bands2 = np.arange(3*3*3*2).reshape(3, 3, 3, 2)
        eop2.data['bands'] = bands2

        eop = data_OOI.OOI.concatenate(eop1, eop2)
        self.assertTrue(np.array_equal(eop.data['bands'], 
                                       np.concatenate((bands1, bands2), axis=0)),
                            msg="Array mismatch")

    def test_concatenate_different_key(self):
        eop1 = data_OOI.OOI()
        bands1 = np.arange(2*3*3*2).reshape(2, 3, 3, 2)
        eop1.data['bands'] = bands1

        eop2 = data_OOI.OOI()
        bands2 = np.arange(3*3*3*2).reshape(3, 3, 3, 2)
        eop2.data['measurements'] = bands2
        
        eop = data_OOI.OOI.concatenate(eop1, eop2)
        self.assertTrue('bands' in eop.data and 'measurements' in eop.data,
                         'Failed to concatenate different data')

    def test_concatenate_timeless(self):
        eop1 = data_OOI.OOI()
        mask1 = np.arange(3*3*2).reshape(3, 3, 2)
        eop1.data_timeless['mask1'] = mask1
        eop1.data_timeless['mask'] = 5 * mask1

        eop2 = data_OOI.OOI()
        mask2 = np.arange(3*3*2).reshape(3, 3, 2)
        eop2.data_timeless['mask2'] = mask2
        eop2.data_timeless['mask'] = 5 * mask1  # add mask1 to eop2

        eop = data_OOI.OOI.concatenate(eop1, eop2)

        for name in ['mask', 'mask1', 'mask2']:
            self.assertTrue(name in eop.data_timeless)
        self.assertTrue(np.array_equal(eop.data_timeless['mask'], 5 * mask1), 
                        "Data with same values should stay the same")

    def test_concatenate_missmatched_timeless(self):
        mask = np.arange(3*3*2).reshape(3, 3, 2)

        eop1 = data_OOI.OOI()
        eop1.data_timeless['mask'] = mask
        eop1.data_timeless['nask'] = 3 * mask

        eop2 = data_OOI.OOI()
        eop2.data_timeless['mask'] = mask
        eop2.data_timeless['nask'] = 5 * mask

        with self.assertRaises(ValueError):
            _ = data_OOI.OOI.concatenate(eop1, eop2)

    def test_equals(self):
        eop1 = data_OOI.OOI(data={'bands': np.arange(2 * 3 * 3 * 2, 
                                        dtype=np.float32).reshape(2, 3, 3, 2)})
        eop2 = data_OOI.OOI(data={'bands': np.arange(2 * 3 * 3 * 2, 
                                        dtype=np.float32).reshape(2, 3, 3, 2)})
        self.assertEqual(eop1, eop2)

        eop1.data['bands'][1, ...] = np.nan
        self.assertNotEqual(eop1, eop2)

        eop2.data['bands'][1, ...] = np.nan
        self.assertEqual(eop1, eop2)

        eop1.data['bands'] = np.reshape(eop1.data['bands'], (2, 3, 2, 3))
        self.assertNotEqual(eop1, eop2)

        eop2.data['bands'] = np.reshape(eop2.data['bands'], (2, 3, 2, 3))
        eop1.data['bands'] = eop1.data['bands'].astype(np.float16)
        self.assertNotEqual(eop1, eop2)

        del eop1.data['bands']
        del eop2.data['bands']
        self.assertEqual(eop1, eop2)

        eop1.data_timeless['dem'] = np.arange(3 * 3 * 2).reshape(3, 3, 2)

        self.assertNotEqual(eop1, eop2)

    def test_timestamp_consolidation(self):
        # 10 frames
        timestamps = [datetime.datetime(2017, 1, 1, 10, 4, 7),
                      datetime.datetime(2017, 1, 4, 10, 14, 5),
                      datetime.datetime(2017, 1, 11, 10, 3, 51),
                      datetime.datetime(2017, 1, 14, 10, 13, 46),
                      datetime.datetime(2017, 1, 24, 10, 14, 7),
                      datetime.datetime(2017, 2, 10, 10, 1, 32),
                      datetime.datetime(2017, 2, 20, 10, 6, 35),
                      datetime.datetime(2017, 3, 2, 10, 0, 20),
                      datetime.datetime(2017, 3, 12, 10, 7, 6),
                      datetime.datetime(2017, 3, 15, 10, 12, 14)]

        data = np.random.rand(10, 100, 100, 3)
        mask = np.random.randint(0, 2, (10, 100, 100, 1))
        mask_timeless = np.random.randint(10, 20, (100, 100, 1))
        scalar = np.random.rand(10, 1)

        eop = data_OOI.OOI(timestamp=timestamps,
                      data={'DATA': data},
                      mask={'MASK': mask},
                      scalar={'SCALAR': scalar},
                      mask_timeless={'MASK_TIMELESS': mask_timeless})

        good_timestamps = timestamps.copy()
        del good_timestamps[0]
        del good_timestamps[-1]
        good_timestamps.append(datetime.datetime(2017, 12, 1))

        removed_frames = eop.consolidate_timestamps(good_timestamps)

        self.assertEqual(good_timestamps[:-1], eop.timestamp)
        self.assertEqual(len(removed_frames), 2)
        self.assertTrue(timestamps[0] in removed_frames)
        self.assertTrue(timestamps[-1] in removed_frames)
        self.assertTrue(np.array_equal(data[1:-1, ...], eop.data['DATA']))
        self.assertTrue(np.array_equal(mask[1:-1, ...], eop.mask['MASK']))
        self.assertTrue(np.array_equal(scalar[1:-1, ...], eop.scalar['SCALAR']))
        self.assertTrue(np.array_equal(mask_timeless, eop.mask_timeless['MASK_TIMELESS']))



if __name__ == '__main__':
    unittest.main()



