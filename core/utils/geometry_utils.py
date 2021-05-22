# -*- coding: utf-8 -*-
"""
Created on Sat May 22 16:25:24 2021

@author: freeridingeo
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import MultiLineString, MultiPolygon
from shapely.wkt import loads

def get_shape_geometry_coords(geometry):
    if geometry.type == 'LineString':
        return list(geometry.coords)
    elif geometry.type == 'MultiLineString':
        coords = []
        for line in geometry:
            coords += list(line.coords)
        return coords

def boundingBoxToOffsets(bbox, geotransform):
    col1 = int((bbox[0] - geotransform[0]) / geotransform[1])
    col2 = int((bbox[1] - geotransform[0]) / geotransform[1]) + 1
    row1 = int((bbox[3] - geotransform[3]) / geotransform[5])
    row2 = int((bbox[2] - geotransform[3]) / geotransform[5]) + 1
    return [row1, row2, col1, col2]

def _split_multigeom_row(gdf_row, geom_col):
    new_rows = []
    if isinstance(gdf_row[geom_col], MultiPolygon) \
            or isinstance(gdf_row[geom_col], MultiLineString):
        new_polys = _split_multigeom(gdf_row[geom_col])
        for poly in new_polys:
            row_w_poly = gdf_row.copy()
            row_w_poly[geom_col] = poly
            new_rows.append(row_w_poly)
    return pd.DataFrame(new_rows)

def _split_multigeom(multigeom):
    return list(multigeom)

def split_multi_geometries(gdf, obj_id_col=None, group_col=None,
                           geom_col='geometry'):
    """Split apart MultiPolygon or MultiLineString geometries.
    
    Arguments
    ---------
    gdf : :class:`geopandas.GeoDataFrame` or `str`
        A :class:`geopandas.GeoDataFrame` or path to a geojson containing
        geometries.
    obj_id_col : str, optional
        If one exists, the name of the column that uniquely identifies each
        geometry (e.g. the ``"BuildingId"`` column in many SpaceNet datasets).
        This will be tracked so multiple objects don't get produced with
        the same ID. Note that object ID column will be renumbered on output.
        If passed, `group_col` must also be provided.
    group_col : str, optional
        A column to identify groups for sequential numbering (for example,
        ``'ImageId'`` for sequential number of ``'BuildingId'``). Must be
        provided if `obj_id_col` is passed.
    geom_col : str, optional
        The name of the column in `gdf` that corresponds to geometry. Defaults
        to ``'geometry'``.

    Returns
    -------
    :class:`geopandas.GeoDataFrame`
        A `geopandas.GeoDataFrame` that's identical to the input, except with
        the multipolygons split into separate rows, and the object ID column
        renumbered (if one exists).
    """
    if obj_id_col and not group_col:
        raise ValueError('group_col must be provided if obj_id_col is used.')

    # drop duplicate columns (happens if loading a csv with geopandas)
    gdf = gdf.loc[:, ~gdf.columns.duplicated()]
    if len(gdf) == 0:
        return gdf

   # check if the values in gdf[geometry] are polygons; if strings, do loads
    if isinstance(gdf[geom_col].iloc[0], str):
        gdf[geom_col] = gdf[geom_col].apply(loads)
    split_geoms_gdf = pd.concat(
        gdf.apply(_split_multigeom_row, axis=1, geom_col=geom_col).tolist())
    gdf = gdf.drop(index=split_geoms_gdf.index.unique())  # remove multipolygons
    gdf = gpd.GeoDataFrame(pd.concat([gdf, split_geoms_gdf],
                                      ignore_index=True), crs=gdf.crs)

    if obj_id_col:
        gdf[obj_id_col] = gdf.groupby(group_col).cumcount()+1

    return gdf




    