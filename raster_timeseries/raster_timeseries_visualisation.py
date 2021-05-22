# -*- coding: utf-8 -*-
"""
Created on Fri May 14 10:27:34 2021

@author: freeridingeo
"""

import sys
import numpy as np

sys.path.append("D:/Code/eotopia/IO")
from rasterio_IO import save_multibandraster_to_eopatch as saveit
import imageio


def create_rastertimeseries_gif(rasterpath, outpath,
                                dataname="raster_timeseries",
                                duration=4):
    """ 
    rasterpath  str
    outpath     str
    dataname    str
    duration    GIF duration in seconda
                int
    """
    eopatch = saveit(rasterpath, dataname)
    fps = len(eopatch.timestamp)/duration
    
    with imageio.get_writer(outpath, mode='I', fps=fps) as writer:
        for image in eopatch:
            writer.append_data(np.array(image, dtype=np.uint8))

