# -*- coding: utf-8 -*-
"""
Created on Fri May  7 22:11:02 2021

@author: nasta
"""

import os
from pathlib import Path, PurePath

import fs

from sentinelhub import SHConfig

import sys
sys.path.append("D:/Code/eotopia/core")
from data_types import DataType, OverwritePermission, DataFormat

def get_filesystem(path, create=False, config=None, **kwargs):
    """ 
    A utility function for initializing any type of filesystem object with 
    PyFilesystem2 package
    
    :param path: A filesystem path
    :type path: str
    :param create: If the filesystem path doesn't exist this flag indicates 
        to either create it or raise an error
    :type create: bool
    :param config: A configuration object with AWS credentials
    :type config: SHConfig
    :param kwargs: Any keyword arguments to be passed forward
    :return: A filesystem object
    :rtype: fs.FS
    """
    if isinstance(path, Path):
        path = str(path)

    ## TODO!
    if path.startswith('s3://'):
        print("Todo")
#        return load_s3_filesystem(path, config=config, **kwargs)

    return fs.open_fs(path, create=create, **kwargs)

def get_base_filesystem_and_path(*path_parts, **kwargs):
    """ 
    Parses multiple strings that define a filesystem path and returns a 
    filesystem object with a relative path on the filesystem

    :param path_parts: One or more strings defining a filesystem path
    :type path_parts: str
    :param kwargs: Parameters passed to get_filesystem function
    :return: A filesystem object and a relative path
    :rtype: (fs.FS, str)
    """
    path_parts = [str(part) for part in path_parts if part is not None]
    base_path = path_parts[0]

    if '://' in base_path:
        base_path_parts = base_path.split('/', 3)
        filesystem_path = '/'.join(base_path_parts[:-1])
        relative_path = '/'.join([base_path_parts[-1], *path_parts[1:]])

        return get_filesystem(filesystem_path, **kwargs), relative_path

    entire_path = os.path.abspath(os.path.join(*path_parts))
    pure_path = PurePath(entire_path)
    posix_path = pure_path.relative_to(pure_path.anchor).as_posix()
    filesystem_path = base_path.split('\\')[0] if '\\' in base_path else '/'

    return get_filesystem(filesystem_path, **kwargs), posix_path


## TODO!
def load_s3_filesystem(path, strict=False, config=None):
    """ 
    Loads AWS s3 filesystem from a path
    """

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



