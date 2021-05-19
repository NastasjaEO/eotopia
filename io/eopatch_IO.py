# -*- coding: utf-8 -*-
"""
Created on Wed May 19 13:02:23 2021

@author: freeridingeo
"""

from abc import abstractmethod
import fs

import sys
sys.path.append("D:/Code/eotopia/repo_core")
from eodata import EOPatch
from eotask import EOTask
from fs_utils import get_base_filesystem_and_path

class BaseLocalIo(EOTask):
    """ 
    Base abstract class for local IO tasks
    """
    def __init__(self, feature, folder=None, *, image_dtype=None, 
                 no_data_value=0, config=None):
        """
        :param feature: Feature which will be exported or imported
        :type feature: (FeatureType, str)
        :param folder: A directory containing image files or a folder of an image file
        :type folder: str
        :param image_dtype: Type of data to be exported into tiff image or imported from tiff image
        :type image_dtype: numpy.dtype
        :param no_data_value: Value of undefined pixels
        :type no_data_value: int or float
        :param config: A configuration object containing credentials
        :type config: SHConfig
        """
        self.feature = self._parse_features(feature)
        self.folder = folder
        self.image_dtype = image_dtype
        self.no_data_value = no_data_value
        self.config = config

    def _get_filesystem_and_paths(self, filename, timestamps, create_paths=False):
        """ 
        It takes location parameters from init and execute methods, joins 
        them together, and creates a filesystem object and file paths relative 
        to the filesystem object.
        """
        if isinstance(filename, str) or filename is None:
            filesystem, relative_path =\
                get_base_filesystem_and_path(self.folder, filename, config=self.config)
            filename_paths = self._generate_paths(relative_path, timestamps)
        elif isinstance(filename, list):
            filename_paths = []
            for timestamp_index, path in enumerate(filename):
                filesystem, relative_path =\
                    get_base_filesystem_and_path(self.folder, path, config=self.config)
                if len(filename) == len(timestamps):
                    filename_paths.append(*self._generate_paths(relative_path, 
                                            [timestamps[timestamp_index]]))
                elif not timestamps:
                    filename_paths.append(*self._generate_paths(relative_path, timestamps))
                else:
                    raise ValueError('The number of provided timestamps does not match '
                                     'the number of provided filenames.')
        else:
            raise TypeError(f"The 'filename' parameter must either be a\
                            list or a string, but {filename} found")

        if create_paths:
            paths_to_create = {fs.path.dirname(filename_path)\
                               for filename_path in filename_paths}
            for filename_path in paths_to_create:
                filesystem.makedirs(filename_path, recreate=True)

        return filesystem, filename_paths

    @staticmethod
    def _generate_paths(path_template, timestamps):
        """ 
        Uses a filename path template to create a list of actual filename paths
        """
        if not (path_template.endswith('.tif') or path_template.endswith('.tiff')):
            path_template = f'{path_template}.tif'

        if not timestamps:
            return [path_template]

        if '*' in path_template:
            path_template = path_template.replace('*', '%Y%m%dT%H%M%S')

        if timestamps[0].strftime(path_template) == path_template:
            return [path_template]

        return [timestamp.strftime(path_template) for timestamp in timestamps]

    @abstractmethod
    def execute(self, eopatch, **kwargs):
        """ 
        Execute of a base class is not implemented
        """
        raise NotImplementedError




