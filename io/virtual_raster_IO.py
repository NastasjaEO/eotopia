# -*- coding: utf-8 -*-
"""
Created on Sun May 23 14:22:29 2021

@author: freeridingeo
"""

import gdal

def read_vsi_rasterfile(filename):
    """Read text from input <filename:str> using VSI and return <content:str>."""

    vsi_file = gdal.VSIFOpenL(str(filename), str('r'))
    gdal.VSIFSeekL(vsi_file, 0, 2)

    vsi_file_size = gdal.VSIFTellL(vsi_file)
    gdal.VSIFSeekL(vsi_file, 0, 0)

    vsi_file_content = gdal.VSIFReadL(vsi_file_size, 1, vsi_file)
    gdal.VSIFCloseL(vsi_file)
    return str(vsi_file_content.decode())

