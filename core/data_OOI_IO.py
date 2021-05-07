# -*- coding: utf-8 -*-
"""
Created on Fri May  7 20:11:43 2021

@author: nasta
"""

import pickle
import gzip
from collections import defaultdict
import fs
from fs.tempfs import TempFS
import concurrent.futures

import numpy as np
import geopandas as gpd

from sentinelhub.os_utils import sys_is_windows

from .data_types import DataType, DataFormat, OverwritePermission

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
    if overwrite_permission is OverwritePermission.ADD_ONLY or \
        (sys_is_windows()\
         and overwrite_permission is OverwritePermission.OVERWRITE_FEATURES):
        fs_data = list(walk_filesystem(filesystem, patch_location))
    else:
        fs_data = []

#    _check_case_matching(ooi_data, fs_data)

#    if overwrite_permission is OverwritePermission.ADD_ONLY:
#        _check_add_only_permission(ooi_data, fs_data)

#    ftype_folder_map = {(ftype, fs.path.dirname(path))\
#                        for ftype, _, path in ooi_data if not ftype.is_meta()}

#    for ftype, folder in ftype_folder_map:
#        if not filesystem.exists(folder):
#            filesystem.makedirs(folder, recreate=True)

    ## TODO!
#   data_to_save = ((DataIO(filesystem, path),
#                         ooi[(ftype, fname)],
#                         DataFormat.NPY if ftype.is_raster() else DataFormat.PICKLE,
#                         compress_level) for ftype, fname, path in ooi_data)

#    with concurrent.futures.ThreadPoolExecutor() as executor:
        # The following is intentionally wrapped in a list in order to get 
        # back potential exceptions
#        list(executor.map(lambda params: params[0].save(*params[1:]), data_to_save))

def load_ooi(ooi, filesystem, patch_location, data=..., lazy_loading=False):
    """ 
    A utility function used by OOI.load method
    """
    data_list = list(walk_filesystem(filesystem, patch_location, data))
    loading_data = [DataIO(filesystem, path) for _, _, path in data]

    if not lazy_loading:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            loading_data = executor.map(lambda loader: loader.load(), loading_data)

    for (ftype, fname, _), value in zip(data_list, loading_data):
        ooi[(ftype, fname)] = value
    
    return ooi

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


class DataIO:
    """ 
    A class handling saving and loading process of single data at a given location
    """
    def __init__(self, filesystem, path):
        """
        :param filesystem: A filesystem object
        :type filesystem: fs.FS
        :param path: A path in the filesystem
        :type path: str
        """
        self.filesystem = filesystem
        self.path = path

    def __repr__(self):
        """
        A representation method
        """
        return '{}({})'.format(self.__class__.__name__, self.path)

    def load(self):
        """ 
        Method for loading data
        """
        with self.filesystem.openbin(self.path, 'r') as file_handle:
            if self.path.endswith(DataFormat.GZIP.extension()):
                with gzip.open(file_handle, 'rb') as gzip_fp:
                    return self._decode(gzip_fp, self.path)
            return self._decode(file_handle, self.path)

    def save(self, data, file_format, compress_level=0):
        """ 
        Method for saving data
        """
        gz_extension = DataFormat.GZIP.extension() if compress_level else ''
        path = self.path + file_format.extension() + gz_extension

        if isinstance(self.filesystem, (fs.osfs.OSFS, TempFS)):
            with TempFS(temp_dir=self.filesystem.root_path) as tempfs:
                self._save(tempfs, data, 'tmp_data', file_format, compress_level)
                fs.move.move_file(tempfs, 'tmp_data', self.filesystem, path)
            return

    def _save(self, filesystem, data, path, file_format, compress_level=0):
        """ 
        Given a filesystem it saves and compresses the data
        """
        with filesystem.openbin(path, 'w') as file_handle:
            if compress_level == 0:
                self._write_to_file(data, file_handle, file_format)
                return

            with gzip.GzipFile(fileobj=file_handle, 
                               compresslevel=compress_level, 
                               mode='wb') as gzip_file_handle:
                self._write_to_file(data, gzip_file_handle, file_format)

    @staticmethod
    def _write_to_file(data, file, file_format):
        """ 
        Writes to a file
        """
        if file_format is DataFormat.NPY:
            np.save(file, data)
        elif file_format is DataFormat.PICKLE:
            pickle.dump(data, file)

    @staticmethod
    def _decode(file, path):
        """ 
        Loads from a file and decodes content
        """
        if DataFormat.PICKLE.extension() in path:
            data = pickle.load(file)

            # There seems to be an issue in geopandas==0.8.1 
            # where unpickling GeoDataFrames, which were saved with an
            # old geopandas version, loads geometry column into 
            # a pandas.Series instead geopandas.GeoSeries. 
            # Because of that it is missing a crs attribute which is only 
            # attached to the entire GeoDataFrame
            if isinstance(data, gpd.GeoDataFrame) and not\
                isinstance(data.geometry, gpd.GeoSeries):
                data = data.set_geometry('geometry')
            return data

        if DataFormat.NPY.extension() in path:
            return np.load(file)
        raise ValueError('Unsupported data type.')

