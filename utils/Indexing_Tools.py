# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 11:34:15 2021

@author: freeridingeo
"""

import numpy as np
import pandas as pd

def make_longlat_multiindex(lats, longs):
    
    if lats.shape == 2:
        lats = lats.flatten()
        longs = longs.flatten()
    
    arrays = [
            np.round(lats,5),
            np.round(longs,5),
            ]
    tuples = list(zip(*arrays))
    index = pd.MultiIndex.from_tuples(tuples, names=["Latitude", "Longitude"])
    return index    

