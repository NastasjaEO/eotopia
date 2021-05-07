# -*- coding: utf-8 -*-
"""
Created on Fri May  7 22:11:02 2021

@author: nasta
"""

import os
from pathlib import Path, PurePath

import fs

from sentinelhub import SHConfig


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
