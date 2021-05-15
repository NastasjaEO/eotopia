# -*- coding: utf-8 -*-
"""
Created on Sat May 15 10:41:34 2021

@author: freeridingeo
"""

from shapely.geometry import Polygon

def create_dummy_polygon(eopatch, addition_factor):
    """ 
    Creates geometry/polygon if there is no data (at timestamp)

    :param addition_factor: size of the 'blank polygon'
    :type addition_factor: float
    :return: polygon
    :rtype: shapely.geometry.Polygon
    """
    x_blank, y_blank = eopatch.bbox.lower_left
    dummy_geometry = Polygon([[x_blank, y_blank],
                                  [x_blank + addition_factor, y_blank],
                                  [x_blank + addition_factor, y_blank + addition_factor],
                                  [x_blank, y_blank + addition_factor]])
    return dummy_geometry
