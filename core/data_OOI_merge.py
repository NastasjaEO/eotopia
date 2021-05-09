# -*- coding: utf-8 -*-
"""
Created on Sat May  8 16:01:27 2021

@author: nasta
"""

import functools
import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd
from geopandas import GeoDataFrame

import sys
sys.path.append("D:/Code/eotopia/core")
from data_types import DataType
from data_OOI_utils import DataParser

def merge_oois(*oois, data=..., time_dependent_op=None, timeless_op=None):
    """ 
    Merge data of given OOI into a new OOI
    
    :param ooi: Any number of OOIs to be merged together
    :type ooi: OOI
    :param data: A collection of data to be merged together. 
        By default all data will be merged.
    :type data: object
    :param time_dependent_op: An operation to be used to join data for any 
            time-dependent raster data. Before joining time slices of all arrays 
            will be sorted. Supported options are:
        - None (default): If time slices with matching timestamps have the same 
            values, take one. Raise an error otherwise.
        - 'concatenate': Keep all time slices, even the ones with matching timestamps
        - 'min': Join time slices with matching timestamps by taking 
            minimum values. Ignore NaN values.
        - 'max': Join time slices with matching timestamps by taking 
            maximum values. Ignore NaN values.
        - 'mean': Join time slices with matching timestamps by taking 
            mean values. Ignore NaN values.
        - 'median': Join time slices with matching timestamps by taking 
            median values. Ignore NaN values.
    :type time_dependent_op: str or Callable or None
    :param timeless_op: An operation to be used to join data for any timeless 
        raster data. Supported options are:
        - None (default): If arrays are the same, take one. Raise an error otherwise.
        - 'concatenate': Join arrays over the last (i.e. bands) dimension
        - 'min': Join arrays by taking minimum values. Ignore NaN values.
        - 'max': Join arrays by taking maximum values. Ignore NaN values.
        - 'mean': Join arrays by taking mean values. Ignore NaN values.
        - 'median': Join arrays by taking median values. Ignore NaN values.
    :type timeless_op: str or Callable or None
    :return: A dictionary with OOI dataand values
    :rtype: Dict[(DataType, str), object]
    """
    reduce_timestamps = time_dependent_op != 'concatenate'
    time_dependent_op = _parse_operation(time_dependent_op, is_timeless=False)
    timeless_op = _parse_operation(timeless_op, is_timeless=True)

    all_data = {dat for ooi in oois for dat in DataParser(data)(ooi)}
    ooi_content = {}

    timestamps, sort_mask, split_mask = _merge_timestamps(oois, reduce_timestamps)
    ooi_content[DataType.TIMESTAMP] = timestamps

    for dat_item in all_data:
        data_type, ooi_name = dat_item

        if data_type.is_raster():
            if data_type.is_time_dependent():
                ooi_content[dat_item] = _merge_time_dependent_raster_obj(
                    oois, dat_item, time_dependent_op, sort_mask, split_mask)
            else:
                 ooi_content[dat_item] =\
                     _merge_timeless_raster_obj(oois, dat_item,timeless_op)

        if data_type.is_vector():
            ooi_content[dat_item] = _merge_vector_obj(oois, dat_item)

        if data_type is DataType.META_INFO:
            ooi_content[dat_item] = _select_meta_info_data(oois, ooi_name)

        ## TODO!
#        if data_type is DataType.BBOX:
#            ooi_content[dat_item] = _get_common_bbox(oois)

    return ooi_content

def _parse_operation(operation_input, is_timeless):
    """ 
    Transforms operation's instruction (i.e. an input string) into a function 
    that can be applied to a list of arrays. 
    If the input already is a function it returns it.
    """
    if isinstance(operation_input, Callable):
        return operation_input

    try:
        return {
            None: _return_if_equal_operation,
            'concatenate': functools.partial(np.concatenate, 
                  axis=-1 if is_timeless else 0),
            'mean': functools.partial(np.nanmean, axis=0),
            'median': functools.partial(np.nanmedian, axis=0),
            'min': functools.partial(np.nanmin, axis=0),
            'max': functools.partial(np.nanmax, axis=0)
        }[operation_input]
    except KeyError as exception:
        raise ValueError(f'Merge operation {operation_input} is not supported')\
            from exception

def _return_if_equal_operation(arrays):
    """ 
    Checks if arrays are all equal and returns first one of them. 
    If they are not equal it raises an error.
    """
    if _all_equal(arrays):
        return arrays[0]
    raise ValueError('Cannot merge given arrays because their values are not the same')

def _merge_timestamps(oois, reduce_timestamps):
    """ 
    Merges together timestamps from OOIs. 
    It also prepares masks on how to sort and join data in any time-dependent 
    raster object.
    """
    all_timestamps = [timestamp for ooi in oois for timestamp in ooi.timestamp
                      if ooi.timestamp is not None]

    if not all_timestamps:
        return [], None, None

    sort_mask = np.argsort(all_timestamps)
    all_timestamps = sorted(all_timestamps)

    if not reduce_timestamps:
        return all_timestamps, sort_mask, None

    split_mask = [
        index + 1 for index, (timestamp, next_timestamp) in\
            enumerate(zip(all_timestamps[:-1], all_timestamps[1:]))
        if timestamp != next_timestamp
    ]
    reduced_timestamps = [timestamp for index, timestamp in enumerate(all_timestamps)
                          if index == 0 or timestamp != all_timestamps[index - 1]]

    return reduced_timestamps, sort_mask, split_mask

def _merge_time_dependent_raster_obj(oois, data, operation, sort_mask, split_mask):
    """ 
    Merges numpy arrays of a time-dependent raster object with a given operation 
    and masks on how to sort and join time raster's time slices.
    """
    arrays = _extract_data_values(oois, data)

    merged_array = np.concatenate(arrays, axis=0)
    del arrays

    if sort_mask is None:
        return merged_array
    merged_array = merged_array[sort_mask]

    if split_mask is None or len(split_mask) == merged_array.shape[0] - 1:
        return merged_array

    split_arrays = np.split(merged_array, split_mask)
    del merged_array

    try:
        split_arrays = [operation(array_chunk) for array_chunk in split_arrays]
    except ValueError as exception:
        raise ValueError(f'Failed to merge {data} with {operation},\
                         try setting a different value for merging '
                         f'parameter time_dependent_op') from exception

    return np.array(split_arrays)

def _merge_timeless_raster_obj(oois, data, operation):
    """ 
    Merges numpy arrays of a timeless raster object with a given operation.
    """
    arrays = _extract_data_values(oois, data)

    if len(arrays) == 1:
        return arrays[0]

    try:
        return operation(arrays)
    except ValueError as exception:
        raise ValueError(f'Failed to merge {data} with {operation},\
                         try setting a different value for merging '
                         f'parameter timeless_op') from exception

def _merge_vector_obj(oois, data):
    """ 
    Merges GeoDataFrames of a vector obj.
    """
    dataframes = _extract_data_values(oois, data)

    if len(dataframes) == 1:
        return dataframes[0]

    crs_list = [dataframe.crs for dataframe in dataframes if dataframe.crs is not None]
    if not crs_list:
        crs_list = [None]
    if not _all_equal(crs_list):
        raise ValueError(f'Cannot merge data {data} because dataframes are\
                         defined for 'f'different CRS')

    merged_dataframe = GeoDataFrame(pd.concat(dataframes, ignore_index=True), 
                                    crs=crs_list[0])
    merged_dataframe = merged_dataframe.drop_duplicates(ignore_index=True)
    # In future a support for vector operations could be added here

    return merged_dataframe

def _select_meta_info_data(oois, ooi_name):
    """ 
    Selects a value for a meta info data of a merged OOI. 
    By default the value is the first one.
    """
    values = _extract_data_values(oois, (DataType.META_INFO, ooi_name))

    if not _all_equal(values):
        message = f'oois have different values of meta info data {ooi_name}.\
            The first value will be ' \
                  f'used in a merged OOI'
        warnings.warn(message, category=UserWarning)

    return values[0]

def _get_common_bbox(oois):
    """ 
    Makes sure that all OOIs, which define a bounding box and CRS, 
    define the same ones.
    """
    bboxes = [ooi.bbox for ooi in oois if ooi.bbox is not None]

    if not bboxes:
        return None

    if _all_equal(bboxes):
        return bboxes[0]
    raise ValueError('Cannot merge OOIs because they are defined for\
                     different bounding boxes')

def _extract_data_values(oois, data):
    """ 
    A helper function that extracts data values from those OOIs 
    where specific data exists.
    """
    data_type, ooi_name = data
    return [ooi[data] for ooi in oois if ooi_name in ooi[data_type]]

def _all_equal(values):
    """ 
    A helper function that checks if all values in a given list are equal to each other.
    """
    first_value = values[0]

    if isinstance(first_value, np.ndarray):
        is_numeric_dtype = np.issubdtype(first_value.dtype, np.number)
        return all(np.array_equal(first_value, array, equal_nan=is_numeric_dtype)\
                   for array in values[1:])

    return all(first_value == value for value in values[1:])
