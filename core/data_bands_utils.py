# -*- coding: utf-8 -*-
"""
Created on Sat May  8 15:44:51 2021

@author: freeridingeo
"""

import numpy as np


def bgr_to_rgb(bgr):
    """
    Converts Blue, Green, Red to Red, Green, Blue.
    """
    return bgr[..., [2, 1, 0]]

