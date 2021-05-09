# -*- coding: utf-8 -*-
"""
Created on Fri May  7 20:11:43 2021

@author: freeridingeo
"""

from collections import defaultdict
import fs

import numpy as np
import geopandas as gpd

## TODO!
# from .data_types import DataType, DataFormat, OverwritePermission

import sys
sys.path.append("D:/Code/eotopia/core")
from data_types import DataType, OverwritePermission, DataFormat

def walk_main_folder(filesystem, folder_path):
    """ 
    Walks the main OOI folders and yields tuples
    (data type, ooi name, path in filesystem)
    """
    for path in filesystem.listdir(folder_path):
        raw_path = path.split('.')[0].strip('/')

        if '/' in raw_path:
            ftype_str, fname = fs.path.split(raw_path)
        else:
            ftype_str, fname = raw_path, ...

        if DataType.has_value(ftype_str):
            yield DataType(ftype_str), fname, fs.path.combine(folder_path, path)

def walk_data_type_folder(filesystem, folder_path):
    """ 
    Walks a data type subfolder of OOI and yields tuples 
    (ooi name, path in filesystem)
    """
    for path in filesystem.listdir(folder_path):
        if '/' not in path and '.' in path:
            yield path.split('.')[0], fs.path.combine(folder_path, path)


def _check_add_only_permission(ooi_data, filesystem_data):
    """ 
    Checks that no existing data will be overwritten
    """
    filesystem_data = {_to_lowercase(*feature) for feature in filesystem_data}
    ooi_data = {_to_lowercase(*data) for data in ooi_data}

    intersection = filesystem_data.intersection(ooi_data)
    if intersection:
        error_msg = "Cannot save data {} with\
            overwrite_permission=OverwritePermission.ADD_ONLY "
        raise ValueError(error_msg.format(intersection))

def _check_case_matching(ooi_data, filesystem_data):
    """ 
    Checks that no two data in memory or in filesystem differ only by 
    ooi name casing
    """
    lowercase_data = {_to_lowercase(*data) for data in ooi_data}

    if len(lowercase_data) != len(ooi_data):
        raise IOError('Some data differ only in casing and cannot be saved\
                      in separate files.')

    original_data = {(ftype, fname) for ftype, fname, _ in ooi_data}

    for ftype, fname, _ in filesystem_data:
        if (ftype, fname) not in original_data and\
            _to_lowercase(ftype, fname) in lowercase_data:
                raise IOError('There already exists data {} in filesystem\
                              that only differs in casing from the one '
                          'that should be saved'.format((ftype, fname)))

def _to_lowercase(ftype, fname, *_):
    """ 
    Tranforms data to it's lowercase representation
    """
    return ftype, fname if fname is ... else fname.lower()


