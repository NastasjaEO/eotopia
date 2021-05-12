# -*- coding: utf-8 -*-
"""
Created on Fri May  7 22:23:59 2021

@author: freeridingeo
"""

import numpy as np
import geopandas as gpd
from geopandas.testing import assert_geodataframe_equal

def get_common_timestamps(source, target):
    """
    Return indices of timestamps from source that are also found in target.

    :param source: timestamps from source
    :type source: list of datetime objects
    :param target: timestamps from target
    :type target: list of datetime objects
    :return: indices of timestamps from source that are also found in target
    :rtype: list of ints
    """
    remove_from_source = set(source).difference(target)
    remove_from_source_idxs = [source.index(rm_date) for rm_date in remove_from_source]
    return [idx for idx, _ in enumerate(source) if idx not in remove_from_source_idxs]

def deep_eq(fst_obj, snd_obj):
    """
    Compares whether fst_obj and snd_obj are deeply equal.

    In case when both fst_obj and snd_obj are of type np.ndarray or either 
    np.memmap, they are compared using np.array_equal(fst_obj, snd_obj). 
    Otherwise, when they are lists or tuples, they are compared for length and 
    then deep_eq is applied component-wise. 
    When they are dict, they are compared for key set equality, and then deep_eq is
    applied value-wise. 
    For all other data types that are not list, tuple, dict, or np.ndarray, 
    the method falls back to the __eq__ method.
    Because np.ndarray is not a hashable object, it is impossible to form a 
    set of numpy arrays, hence deep_eq works correctly.

    :param fst_obj: First object compared
    :param snd_obj: Second object compared
    :return: `True` if objects are deeply equal, `False` otherwise
    """
    # pylint: disable=too-many-return-statements
    if not isinstance(fst_obj, type(snd_obj)):
        return False

    if isinstance(fst_obj, np.ndarray):
        if fst_obj.dtype != snd_obj.dtype:
            return False
        fst_nan_mask = np.isnan(fst_obj)
        snd_nan_mask = np.isnan(snd_obj)
        return np.array_equal(fst_obj[~fst_nan_mask], snd_obj[~snd_nan_mask]) and \
            np.array_equal(fst_nan_mask, snd_nan_mask)

    if isinstance(fst_obj, gpd.GeoDataFrame):
        try:
            assert_geodataframe_equal(fst_obj, snd_obj)
            return True
        except AssertionError:
            return False

    if isinstance(fst_obj, (list, tuple)):
        if len(fst_obj) != len(snd_obj):
            return False

        for element_fst, element_snd in zip(fst_obj, snd_obj):
            if not deep_eq(element_fst, element_snd):
                return False
        return True

    if isinstance(fst_obj, dict):
        if fst_obj.keys() != snd_obj.keys():
            return False

        for key in fst_obj:
            if not deep_eq(fst_obj[key], snd_obj[key]):
                return False
        return True

    return fst_obj == snd_obj

def constant_pad(X, multiple_of, up_down_rule='even', 
                 left_right_rule='even', pad_value=0):
    """
    Function pads an image of shape (rows, columns, channels) with zeros.
    It pads an image so that the shape becomes 
    (rows + padded_rows, columns + padded_columns, channels), where
    padded_rows = (int(rows/multiple_of[0]) + 1) * multiple_of[0] - rows
    Same rule is applied to columns.
    
    :type X: array of shape (rows, columns, channels) or (rows, columns)
    :param multiple_of: make X' rows and columns multiple of this tuple
    :type multiple_of: tuple (rows, columns)
    :param up_down_rule: Add padded rows evenly to the top/bottom of the image, or up (top) / down (bottom) only
    :type up_down_rule: up_down_rule: string, (even, up, down)
    :param up_down_rule: Add padded columns evenly to the left/right of the image, or left / right only
    :type up_down_rule: up_down_rule: string, (even, left, right)
    :param pad_value: Value to be assigned to padded rows and columns
    :type pad_value: int
    """
    # pylint: disable=invalid-name
    shape = X.shape
    row_padding, col_padding = 0, 0
    if shape[0] % multiple_of[0]:
        row_padding = (int(shape[0] / multiple_of[0]) + 1) * multiple_of[0] - shape[0]

    if shape[1] % multiple_of[1]:
        col_padding = (int(shape[1] / multiple_of[1]) + 1) * multiple_of[1] - shape[1]

    row_padding_up, row_padding_down, col_padding_left, col_padding_right = 0, 0, 0, 0

    if row_padding > 0:
        if up_down_rule == 'up':
            row_padding_up = row_padding
        elif up_down_rule == 'down':
            row_padding_down = row_padding
        elif up_down_rule == 'even':
            row_padding_up = int(row_padding / 2)
            row_padding_down = row_padding_up + (row_padding % 2)
        else:
            raise ValueError('Padding rule for rows not supported. Choose beteen even, down or up!')

    if col_padding > 0:
        if left_right_rule == 'left':
            col_padding_left = col_padding
        elif left_right_rule == 'right':
            col_padding_right = col_padding
        elif left_right_rule == 'even':
            col_padding_left = int(col_padding / 2)
            col_padding_right = col_padding_left + (col_padding % 2)
        else:
            raise ValueError('Padding rule for columns not supported.\
                             Choose beteen even, left or right!')

    return np.lib.pad(X, ((row_padding_up, row_padding_down), 
                          (col_padding_left, col_padding_right)),
                          'constant', constant_values=((pad_value, pad_value), 
                                                       (pad_value, pad_value)))


