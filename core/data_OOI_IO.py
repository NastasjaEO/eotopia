# -*- coding: utf-8 -*-
"""
Created on Fri May  7 20:11:43 2021

@author: nasta
"""

from collections import defaultdict
import fs
import concurrent.futures

import geopandas as gpd

from .data_types import DataType, OverwritePermission

def save_ooi(ooi, filesystem, patch_location, data=..., 
             overwrite_permission=OverwritePermission.ADD_ONLY, 
             compress_level=0):
    """ 
    A utility function used by OOI.save method
    """
    patch_exists = filesystem.exists(patch_location)

    if overwrite_permission is OverwritePermission.OVERWRITE_PATCH and patch_exists:
        filesystem.removetree(patch_location)
        if patch_location != '/':
            patch_exists = False

    if not patch_exists:
        filesystem.makedirs(patch_location, recreate=True)

    ## TODO!
#    ooi_data = list(walk_ooi(ooi, patch_location, data))

    ## TODO!
#    if overwrite_permission is OverwritePermission.ADD_ONLY or \
#            (sys_is_windows() and overwrite_permission is OverwritePermission.OVERWRITE_FEATURES):
#        fs_features = list(walk_filesystem(filesystem, patch_location))
#    else:
#        fs_features = []

#    _check_case_matching(ooi_data, fs_features)


#    if overwrite_permission is OverwritePermission.ADD_ONLY:
#        _check_add_only_permission(ooi_data, fs_features)

#    ftype_folder_map = {(ftype, fs.path.dirname(path))\
#                        for ftype, _, path in ooi_data if not ftype.is_meta()}

#    for ftype, folder in ftype_folder_map:
#        if not filesystem.exists(folder):
#            filesystem.makedirs(folder, recreate=True)

    ## TODO!
#   data_to_save = ((FeatureIO(filesystem, path),
#                         ooi[(ftype, fname)],
#                         FileFormat.NPY if ftype.is_raster() else FileFormat.PICKLE,
#                         compress_level) for ftype, fname, path in ooi_data)

#    with concurrent.futures.ThreadPoolExecutor() as executor:
        # The following is intentionally wrapped in a list in order to get 
        # back potential exceptions
#        list(executor.map(lambda params: params[0].save(*params[1:]), data_to_save))

def load_ooi(ooi, filesystem, patch_location, data=..., lazy_loading=False):
    """ 
    A utility function used by OOI.load method
    """
    ## TODO!
#    features = list(walk_filesystem(filesystem, patch_location, data))
#    loading_data = [FeatureIO(filesystem, path) for _, _, path in data]

#    if not lazy_loading:
#        with concurrent.futures.ThreadPoolExecutor() as executor:
#            loading_data = executor.map(lambda loader: loader.load(), loading_data)

#    for (ftype, fname, _), value in zip(data, loading_data):
 #       ooi[(ftype, fname)] = value

#    return ooi

def walk_filesystem(filesystem, patch_location, data=...):
    """ 
    Recursively reads a patch_location and returns yields tuples of 
    (data_type, ooi_name, file_path)
    """
    existing_data = defaultdict(dict)

    for ftype, fname, path in walk_main_folder(filesystem, patch_location):
        existing_data[ftype][fname] = path

    returned_meta_data = set()
    queried_data = set()

    ## TODO!
#    for ftype, fname in FeatureParser(data):
#        if fname is ... and not existing_data[ftype]:
#            continue

#        if ftype.is_meta():
#            if ftype in returned_meta_data:
#                continue
#            fname = ...
#            returned_meta_data.add(ftype)

#        elif ftype not in queried_data and (fname is ... or fname not in existing_data[ftype]):
#            queried_data.add(ftype)
#            if ... not in existing_data[ftype]:
#                raise IOError('There are no data of type {} in saved OOI'.format(ftype))

#            for ooi_name, path in walk_feature_type_folder(filesystem, 
                    # existing_data[ftype][...]):
#                existing_data[ftype][ooi_name] = path

#        if fname not in existing_data[ftype]:
#            raise IOError('Data {} does not exist in saved OOI'.format((ftype, fname)))

#        if fname is ... and not ftype.is_meta():
#            for ooi_name, path in existing_data[ftype].items():
#                if ooi_name is not ...:
#                    yield ftype, ooi_name, path
#        else:
#            yield ftype, fname, existing_data[ftype][fname]


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

def walk_ooi(ooi, patch_location, data=...):
    """ 
    Recursively reads a patch_location and returns yields tuples of 
    (data_type, ooi_name, file_path)
    """
    returned_meta_data = set()
    ## TODO!
#    for ftype, fname in FeatureParser(data)(ooi):
#        name_basis = fs.path.combine(patch_location, ftype.value)
#        if ftype.is_meta():
#            if ooi[ftype] and ftype not in returned_meta_data:
#                yield ftype, ..., name_basis
#                returned_meta_data.add(ftype)
#        else:
#            yield ftype, fname, fs.path.combine(name_basis, fname)

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
