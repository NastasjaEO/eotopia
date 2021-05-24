# -*- coding: utf-8 -*-
"""
Created on Mon May 24 11:20:39 2021

@author: freeridingeo
"""

def dictmerge(x, y):
    """
    merge two dictionaries
    """
    z = x.copy()
    z.update(y)
    return z
