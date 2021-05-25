# -*- coding: utf-8 -*-
"""
Created on Sat May 15 10:41:34 2021

@author: freeridingeo
"""

from shapely.geometry import (Polygon, MultiPolygon,
                              Point, MultiPoint, 
                              LineString, MultiLineString)

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

def create_empty_geometry(geom_type, crs, listofcoordinates):
    if geom_type == "Polygon":
        geom = Polygon()
    elif geom_type == "Line":
        geom = LineString()    
    elif geom_type == "Point":
        geom = Point()
    elif geom_type == "MultiPoint":
        geom = MultiPoint()
    elif geom_type == "MultiLine":
        geom = MultiLineString()
    return geom

def create_geometry_from_coordinatelist(geom_type, crs, listofcoordinates):
    if geom_type == "Polygon":
        geom = Polygon(listofcoordinates)
    elif geom_type == "Line":
        geom = LineString(listofcoordinates)    
    elif geom_type == "Point":
        length = len(listofcoordinates)
        if length == 1:
            geom = Point(listofcoordinates[0])
        elif length > 1:
            geom = []
            for i in listofcoordinates:
                pt = Point(i[0], i[1])
                geom.append(pt)
    elif geom_type == "MultiPoint" and len(listofcoordinates) > 1:
        geom = MultiPoint(listofcoordinates)
    elif geom_type == "MultiLine"and len(listofcoordinates) > 1:
        geom = MultiLineString([listofcoordinates])

    return geom

def create_linestring_from_points(points):
    geom = LineString(points)
    return geom