# -*- coding: utf-8 -*-
"""
Created on Thu May 13 15:15:23 2021

@author: freeridingeo
"""

import numpy as np
import rasterio

def return_pixel_index(pixel_coordinates, window):
    """
    pixel_coordinates: Tuple
    window: Window
    Return indexes of pixel mapping for raster window.
    """
    (row_min, row_max), (col_min, col_max) = window.toranges()

    index_window = np.logical_and.reduce(
        (
            pixel_coordinates[0] >= row_min,
            pixel_coordinates[0] < row_max,
            pixel_coordinates[1] >= col_min,
            pixel_coordinates[1] < col_max,
        )
    )
    return index_window

def raster_bandvalue_histogram(raster, bins=50):
    rasterio.plot.show_hist(raster, bins=bins, lw=0.0, 
                            stacked=False, alpha=0.3,
                            histtype='stepfilled', title="Histogram")

def rasterize(vectorobject, reference, outname=None, burn_values=1, 
              expressions=None, nodata=0, append=False):
    """
    rasterize a vector object
    Parameters
    ----------
    vectorobject: Vector
        the vector object to be rasterized
    reference: Raster
        a reference Raster object to retrieve geo information and extent from
    outname: str or None
        the name of the GeoTiff output file; if None, an in-memory object of type :class:`Raster` is returned and
        parameter outname is ignored
    burn_values: int or list
        the values to be written to the raster file
    expressions: list
        SQL expressions to filter the vector object by attributes
    nodata: int
        the nodata value of the target raster file
    append: bool
        if the output file already exists, update this file with new rasterized values?
        If True and the output file exists, parameters `reference` and `nodata` are ignored.
    Returns
    -------
    Raster or None
        if outname is `None`, a raster object pointing to an in-memory dataset else `None`
    Example
    -------
    >>> from spatialist import Vector, Raster, rasterize
    >>> outname1 = 'target1.tif'
    >>> outname2 = 'target2.tif'
    >>> with Vector('source.shp') as vec:
    >>>     with Raster('reference.tif') as ref:
    >>>         burn_values = [1, 2]
    >>>         expressions = ['ATTRIBUTE=1', 'ATTRIBUTE=2']
    >>>         rasterize(vec, reference, outname1, burn_values, expressions)
    >>>         expressions = ["ATTRIBUTE2='a'", "ATTRIBUTE2='b'"]
    >>>         rasterize(vec, reference, outname2, burn_values, expressions)
    """

