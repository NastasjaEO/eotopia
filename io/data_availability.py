# -*- coding: utf-8 -*-
"""
Created on Fri May 21 10:49:51 2021

@author: freeridingeo
"""

from sentinelhub import DataCollection

sentinelhub_abbrev = {
    'S2_L1C': DataCollection.SENTINEL2_L1C,
    'S2_L2A' : DataCollection.SENTINEL2_L2A,
    'S1' : DataCollection.SENTINEL1,
    'S1_IW' : DataCollection.SENTINEL1_IW,
    'S1_IW_A' : DataCollection.SENTINEL1_IW_ASC,
    'S1_IW_D' : DataCollection.SENTINEL1_IW_DES,
    'S1_EW' : DataCollection.SENTINEL1_EW,
    'S1_EW_A' : DataCollection.SENTINEL1_EW_ASC,
    'S1_EW_D' : DataCollection.SENTINEL1_EW_DES,
    'S1_EW_SH' : DataCollection.SENTINEL1_EW_SH,
    'S1_EW_SH_A' : DataCollection.SENTINEL1_EW_SH_ASC,
    'S1_EW_SH_D' : DataCollection.SENTINEL1_EW_SH_DES,
    'DEM' : DataCollection.DEM,
    'MODIS' : DataCollection.MODIS,
    'L8' : DataCollection.LANDSAT8,
    'S5' : DataCollection.SENTINEL5P,
    'S3_O' : DataCollection.SENTINEL3_OLCI,
    'S3_S' : DataCollection.SENTINEL3_SLSTR,
    }

def list_sentinelhub_data():
    for collection in DataCollection.get_available_collections():
        print(collection)

    