# -*- coding: utf-8 -*-
"""
Created on Tue May 18 10:53:18 2021

@author: freeridingeo
"""

import numpy as np

def get_first_and_last_date_of_eopatches(eopatches):

    eopatches = np.array(eopatches)

    timelist = []
    for eopatch in eopatches:
        timelist.append(eopatch.timestamp[0])
    mindate = str(max(timelist).date())
    print('Earliest date: ' + str(max(timelist)))

    timelist = []
    for eopatch in eopatches:
        timelist.append(eopatch.timestamp[-1])
    maxdate = str(min(timelist).date())
    print('Latest date: ' + str(min(timelist)))
    return mindate, maxdate