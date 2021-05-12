# -*- coding: utf-8 -*-
"""
Created on Thu May  6 19:38:36 2021

@author: freeridingeo
"""

import attr
import copy
import logging
import warnings
import pickle
import gzip
import fs
from fs.tempfs import TempFS
import concurrent.futures

from collections import OrderedDict, defaultdict

from sentinelhub.os_utils import sys_is_windows

import numpy as np
import geopandas as gpd

import dateutil.parser
import datetime

from sentinelhub import BBox, CRS

## TODO!
#from .data_types import DataType, OverwritePermission

import sys
sys.path.append("D:/Code/eotopia/core")
#from data_OOI_utils import ()
from data_types import DataType, OverwritePermission, DataFormat
from data_OOI_utils import deep_eq
from data_OOI_merge import (_parse_operation, _select_meta_info_data,
                            _merge_timestamps, _merge_time_dependent_raster_obj,
                            _merge_timeless_raster_obj, _merge_vector_obj)

sys.path.append("D:/Code/eotopia/utils")
from filesystem_utils import (get_filesystem, 
                              walk_data_type_folder, walk_main_folder,
                              _check_add_only_permission, _check_case_matching)

LOGGER = logging.getLogger(__name__)
warnings.simplefilter('default', DeprecationWarning)

MAX_DATA_REPR_LEN = 100


@attr.s(repr=False, eq=False, kw_only=True)
class OOI:
    """
    Object Of Interest
    Usually an EO data thingy
    """

    num_data = attr.ib(factory=dict)
    mask = attr.ib(factory=dict)
    scalar = attr.ib(factory=dict)
    label = attr.ib(factory=dict)
    vector = attr.ib(factory=dict)
    data_timeless = attr.ib(factory=dict)
    mask_timeless = attr.ib(factory=dict)
    scalar_timeless = attr.ib(factory=dict)
    label_timeless = attr.ib(factory=dict)
    vector_timeless = attr.ib(factory=dict)
    meta_info = attr.ib(factory=dict)
    bbox = attr.ib(default=None)
    timestamp = attr.ib(factory=list)
    
    def __setattr__(self, key, value, ooi_name=None):
        """
        Raises TypeError if data type attributes are not of correct type.
        In case they are a dictionary they are cast to _DataDict class
        """
        if ooi_name not in (None, Ellipsis) and DataType.has_value(key):
            self[key][ooi_name] = value
            return

        if DataType.has_value(key) and not isinstance(value, DataIO):
            data_type = DataType(key)
            value = self._parse_data_type_value(data_type, value)
        super().__setattr__(key, value)

    @staticmethod
    def _parse_data_type_value(data_type, value):
        """ 
        Checks or parses value which will be assigned to a data type 
        attribute of `OOI`. 
        If the value cannot be parsed correctly it raises an error.
        :raises: TypeError, ValueError
        """
        if data_type.has_dict() and isinstance(value, dict):
            return value if isinstance(value, _DataDict) else _DataDict(value, 
                                                                        data_type)

        if data_type is DataType.BBOX:
            if value is None or isinstance(value, BBox):
                return value
            if isinstance(value, (tuple, list)) and len(value) == 5:
                return BBox(value[:4], crs=value[4])

        if data_type is DataType.TIMESTAMP:
            if isinstance(value, (tuple, list)):
                return [timestamp\
                        if isinstance(timestamp, datetime.date)\
                            else dateutil.parser.parse(timestamp)
                        for timestamp in value]

        raise TypeError('Attribute {} requires value of type {} - '
                        'failed to parse given value {}'.\
                            format(data_type, data_type.type(), value))

    def __getattribute__(self, key, load=True, ooi_name=None):
        value = super().__getattribute__(key)

        if isinstance(value, DataIO) and load:
            value = value.load()
            setattr(self, key, value)
            value = getattr(self, key)

        if ooi_name not in (None, Ellipsis) and isinstance(value, _DataDict):
            return value[ooi_name]

        return value

    def __getitem__(self, data_type):
        """ 
        Provides features of requested data type. 
        It can also accept a tuple of (data_type, ooi_name)

        :param data_type: Type of OOI data
        :type data_type: DataType or str or (DataType, str)
        :return: Dictionary of features
        """
        ooi_name = None

        if isinstance(data_type, tuple):
            self._check_tuple_key(data_type)
            data_type, ooi_name = data_type

        return self.__getattribute__(DataType(data_type).value, 
                                     ooi_name=ooi_name)

    def __setitem__(self, data_type, value):
        """ 
        Sets a new dictionary / list to the given DataType. 
        As a key it can also accept a tuple of (data_type, ooi_name)

        :param data_type: Type of OOI data
        :type data_type: DataType or str or (DataType, str)
        :param value: New dictionary or list
        :type value: dict or list
        :return: Dictionary of ooi features
        """
        ooi_name = None

        if isinstance(data_type, tuple):
            self._check_tuple_key(data_type)
            data_type, ooi_name = data_type

        return self.__setattr__(DataType(data_type).value, value, 
                                ooi_name=ooi_name)

    @staticmethod
    def _check_tuple_key(key):
        """ 
        A helper function that checks a tuple, which should hold 
        (data_type, ooi_name)
        """
        if len(key) != 2:
            raise ValueError('Given element should be a data_type or a '
                             'tuple of (data_type, ooi_name),'
                             'but {} found'.format(key))        

    def __eq__(self, other):
        """
        True if DataType attributes, bbox, and timestamps of both oois 
        are equal by value.
        """
        if not isinstance(self, type(other)):
            return False

        for data_type in DataType:
            if not deep_eq(self[data_type], other[data_type]):
                return False
        return True

    def __add__(self, other):
        """ 
        Merges two oois into a new ooi
        """
        return self.merge(other)

    def __repr__(self):
        data_repr_list = ['{}('.format(self.__class__.__name__)]
        for data_type in DataType:
            content = self[data_type]

            if isinstance(content, dict) and content:
                content_str = '\n    '.\
                    join(['{'] + ['{}: {}'.format(label, 
                                                  self._repr_value(value))\
                                          for label, value in
                                                     sorted(content.items())]) + '\n  }'
            else:
                content_str = self._repr_value(content)
            data_repr_list.append('{}: {}'.format(data_type.value, content_str))
        return '\n  '.join(data_repr_list) + '\n)'

    @staticmethod
    def _repr_value(value):
        """Creates a representation string for different types of data.

        :param value: data in any type
        :return: representation string
        :rtype: str
        """
        if isinstance(value, np.ndarray):
            return '{}(shape={}, dtype={})'.format(OOI._repr_value_class(value), 
                                                   value.shape, value.dtype)

        if isinstance(value, gpd.GeoDataFrame):
            crs = CRS(value.crs).ogc_string() if value.crs else value.crs
            return f'{OOI._repr_value_class(value)}(' \
                   f'columns={list(value)}, ' \
                   f'length={len(value)}, ' \
                   f'crs={crs})'

        if isinstance(value, (list, tuple, dict)) and value:
            repr_str = str(value)
            if len(repr_str) <= MAX_DATA_REPR_LEN:
                return repr_str

            bracket_str = '[{}]' if isinstance(value, list) else '({})'
            if isinstance(value, (list, tuple)) and len(value) > 2:
                repr_str = bracket_str.format('{}, ..., {}'.\
                                        format(repr(value[0]), repr(value[-1])))

            if len(repr_str) > MAX_DATA_REPR_LEN and isinstance(value, (list, tuple)) and len(value) > 1:
                repr_str = bracket_str.format('{}, ...'.format(repr(value[0])))

            if len(repr_str) > MAX_DATA_REPR_LEN:
                repr_str = str(type(value))

            return '{}, length={}'.format(repr_str, len(value))
        return repr(value)

    @staticmethod
    def _repr_value_class(value):
        """ 
        A representation of a class of a given value
        """
        cls = value.__class__
        return '.'.join([cls.__module__.split('.')[0], cls.__name__])

    def __copy__(self, data=...):
        """
        Returns a new OOI with shallow copies of given data.
        :param features: A collection of data or data types that will be 
            copied into new OOI.
        :type features: object supported by eolearn.core.utilities.FeatureParser class
        """
        if not data:  # For some reason deepcopy and copy pass {} by default
            data = ...

        new_ooi = OOI()
        for data_type, ooi_name in DataParser(data)(self):
            if ooi_name is ...:
                new_ooi[data_type] = copy.copy(self[data_type])
            else:
                new_ooi[data_type][ooi_name] = self[data_type][ooi_name]
        return new_ooi

    def __deepcopy__(self, memo=None, data=...):
        """
        Returns a new OOI with deep copies of given data.
        :param memo: built-in parameter for memoization
        :type memo: dict
        :param data: A collection of data or data types that will be copied 
            into new OOI.
        :type data: object supported by FeatureParser class
        """
        if not data:  # For some reason deepcopy and copy pass {} by default
            data = ...

        new_ooi = self.__copy__(data=data)
        for datatype in DataType:
            new_ooi[datatype] = copy.deepcopy(new_ooi[datatype], memo)

        return new_ooi

    def remove_feature(self, data_type, ooi_name):
        """
        Removes the data item 'ooi_name'from dictionary of 'data_type'.

        :param data_type: Enum of the attribute we're about to modify
        :type data_type: DataType
        :param ooi_name: Name of the ooi of the attribute
        :type ooi_name: str
        """
        LOGGER.debug("Removing feature '%s' from attribute '%s'", 
                     ooi_name, data_type.value)

        if ooi_name in self[data_type]:
            del self[data_type][ooi_name]

    def add_feature(self, data_type, ooi_name, value):
        """
        Sets OOI[data_type][ooi_name] to the given value.

        :param data_type: Enum of the attribute we're about to modify
        :type data_type: DataType
        :param ooi_name: Name of the ooi of the attribute
        :type ooi_name: str
        :param value: New value of the feature
        :type value: object

        """
        LOGGER.debug("Adding to data '%s' value '%s'", 
                     ooi_name, value)

        self[data_type][ooi_name] = value

        ## TODO!
        # def rename_feature(self, data_type, ooi_name, new_ooi_name):

    @staticmethod
    def _check_if_dict(data_type):
        """
        Checks if the given data type contains a dictionary and raises an 
        error if it doesn't.
        
        :param data_type: Enum of the attribute we're about to modify
        :type data_type: DataType
        :raise: TypeError
        """
        data_type = DataType(data_type)
        if data_type.type() is not dict:
            raise TypeError('{} does not contain a dictionary of features'.\
                            format(data_type))

    def reset_data_type(self, data_type):
        """
        Resets the values of the given data type.
        
        :param data_type: Type of a data
        :type data_type: DataType
        """
        data_type = DataType(data_type)
        if data_type.has_dict():
            self[data_type] = {}
        elif data_type is DataType.BBOX:
            self[data_type] = None
        else:
            self[data_type] = []

    def set_bbox(self, new_bbox):
        """
        :param new_bbox: new bbox
        :type: new_bbox: BBox
        """
        self.bbox = new_bbox

    def set_timestamp(self, new_timestamp):
        """
        :param new_timestamp: list of dates
        :type new_timestamp: list(str)
        """
        self.timestamp = new_timestamp

    def get_feature(self, data_type, ooi_name=None):
        """
        Returns the array of corresponding data item.
        
        :param data_type: Enum of the attribute we're about to modify
        :type data_type: DataType
        :param ooi_name: Name of the ooi of the attribute
        :type ooi_name: str
        """
        if ooi_name is None:
            return self[data_type]
        return self[data_type][ooi_name]

    def get_features(self):
        """
        Returns a dictionary of all non-empty features of OOI.
        The elements are either sets of ooi names or a boolean `True` in case 
        data type has no dictionary of ooi names.

        :return: A dictionary of features
        :rtype: dict(DataType: str or True)
        """
        data_dict = {}
        for data_type in DataType:
            if self[data_type]:
                data_dict[data_type] = set(self[data_type]) \
                    if data_type.has_dict() else True

        return data_dict

    def get_spatial_dimension(self, data_type, ooi_name):
        """
        Returns a tuple of spatial dimension (height, width) of a data item.
        The data item has to be spatial or time dependent.
        
        :param data_type: Enum of the attribute we're about to modify
        :type data_type: DataType
        :param ooi_name: Name of the ooi of the attribute
        :type ooi_name: str
        """
        if data_type.is_time_dependent() or data_type.is_spatial():
            shape = self[data_type][ooi_name].shape
            return shape[1:3] if data_type.is_time_dependent() else shape[0:2]

        raise ValueError('FeatureType used to determine the width and height of raster must be'
                         ' time dependent or spatial.')

    def get_ooi_items_list(self):
        """
        Returns a list of all non-empty items of OOI.
        The elements are either only DataType or a pair of DataType and ooi name.

        :return: list of items
        :rtype: list(DataType or (DataType, str))
        """
        item_list = []
        for data_type in DataType:
            if data_type.has_dict():
                for ooi_name in self[data_type]:
                    item_list.append((data_type, ooi_name))
            elif self[data_type]:
                item_list.append(data_type)
        return item_list

    @staticmethod
    def concatenate(ooi1, ooi2):
        """
        Joins all data from two OOIs and returns a new OOI.
        If timestamps don't match it will try to join all time-dependent data 
        with the same name.
        Note: In general the data won't be deep copied. Deep copy will only happen 
        when merging time-dependent data along time
        
        :param ooi1: First OOI
        :type ooi1: OOI
        :param ooi2: Second OOI
        :type eopatch2: OOI
        :return: ooi OOI
        :rtype: OOI
        """
        warnings.warn('OOI.concatenate is deprecated, use a more general '
                      'OOI.merge method instead', DeprecationWarning)

        ooi_content = {}
        timestamps_exist = ooi1.timestamp and ooi2.timestamp
        timestamps_match = timestamps_exist and deep_eq(ooi1.timestamp, 
                                                        ooi2.timestamp)

        for data_type in DataType:

            if data_type.has_dict():
                ooi_content[data_type.value] = {**ooi1[data_type], 
                                                     **ooi2[data_type]}

                for ooi_name in ooi1[data_type].keys() & ooi2[data_type].keys():
                    data1 = ooi1[data_type][ooi_name]
                    data2 = ooi2[data_type][ooi_name]

                    if data_type.is_time_dependent() and not timestamps_match:
                        ooi_content[data_type.value][ooi_name] =\
                            OOI.concatenate_data(data1, data2)
                    elif not deep_eq(data1, data2):
                        raise ValueError('Could not merge ({}, {}) feature' 
                                         'because values differ'.\
                                             format(data_type, ooi_name))

            elif data_type is DataType.TIMESTAMP and\
                    timestamps_exist and not timestamps_match:
                ooi_content[data_type.value] = ooi1[data_type] + ooi2[data_type]
            else:
                if not ooi1[data_type] or deep_eq(ooi1[data_type], ooi2[data_type]):
                    ooi_content[data_type.value] = copy.copy(ooi2[data_type])
                elif not ooi2[data_type]:
                    ooi_content[data_type.value] = copy.copy(ooi1[data_type])
                else:
                    raise ValueError('Could not merge {} feature because' 
                                     'values differ'.format(data_type))

        return OOI(**ooi_content)

    @staticmethod
    def concatenate_data(data1, data2):
        """
        A method that concatenates two numpy array along first axis.
        
        :param data1: Numpy array of shape (times1, height, width, n_features)
        :type data1: numpy.ndarray
        :param data2: Numpy array of shape (times2, height, width, n_features)
        :type data1: numpy.ndarray
        :return: Numpy array of shape (times1 + times2, height, width, n_features)
        :rtype: numpy.ndarray
        """
        if data1.shape[1:] != data2.shape[1:]:
            raise ValueError('Could not concatenate data because non-temporal' 
                             'dimensions do not match')
        return np.concatenate((data1, data2), axis=0)
 
    def save(self, path, data=..., 
             overwrite_permission=OverwritePermission.ADD_ONLY, 
             compress_level=0, filesystem=None):
        """ 
        Method to save an OOI from memory to a storage
        The output is a folder with the data stored as .npy

        :param path: A location where to save OOI. 
            It can be either a local path or a remote URL path.
        :type path: str
        :param data: A collection of data types specifying data of which type 
            will be saved. By default all features will be saved.
        :type data: list(DataType) or list((DataType, str)) or ...
        :param overwrite_permission: A level of permission for overwriting 
            an existing OOI
        :type overwrite_permission: OverwritePermission or int
        :param compress_level: A level of data compression and can be specified 
            with an integer from 0 (no compression)to 9 (highest compression).
        :type compress_level: int
        :param filesystem: An existing filesystem object. If not given it will 
            be initialized according to the `path` parameter.
        :type filesystem: fs.FS or None
        """
        if filesystem is None:
            filesystem = get_filesystem(path, create=True)
            path = '/'
            
            save_ooi(self, filesystem, path, data=data, 
                        compress_level=compress_level,
                        overwrite_permission=OverwritePermission(overwrite_permission))

    @staticmethod
    def load(path, data=..., lazy_loading=False, filesystem=None):
        """ 
        Method to load an OOI from a storage into memory

        :param path: A location from where to load OOI. 
            It can be either a local path or a remote URL path.
        :type path: str
        :param data: A collection of data to be loaded. By default all data 
            will be loaded.
        :type data: object
        :param lazy_loading: If `True` data will be lazy loaded.
        :type lazy_loading: bool
        :param filesystem: An existing filesystem object. 
            If not given it will be initialized according to the `path`
            parameter.
        :type filesystem: fs.FS or None
        :return: Loaded OOI
        :rtype: OOI
        """
        if filesystem is None:
            filesystem = get_filesystem(path, create=False)
            path = '/'

        return load_ooi(OOI(), filesystem, path, data=data, 
                        lazy_loading=lazy_loading)

    def merge(self, *oois, data=..., 
              time_dependent_op=None, timeless_op=None):
        """ 
        Merge features of given OOIs into a new OOI
        
        :param oois: Any number of OOIs to be merged together with the 
            current OOI
        :type oois: OOI
        :param data: A collection of data to be merged together. 
            By default all data will be merged.
        :type data: object
        :param time_dependent_op: An operation to be used to join data for 
            any time-dependent raster feature. Before joining time slices of 
            all arrays will be sorted. Supported options are:
            - None (default): If time slices with matching timestamps have 
            the same values, take one. Raise an error otherwise.
            - 'concatenate': Keep all time slices, even the ones with matching 
                timestamps
            - 'min': Join time slices with matching timestamps by taking 
                minimum values. Ignore NaN values.
            - 'max': Join time slices with matching timestamps by taking 
                maximum values. Ignore NaN values.
            - 'mean': Join time slices with matching timestamps by taking 
                mean values. Ignore NaN values.
            - 'median': Join time slices with matching timestamps by taking 
                median values. Ignore NaN values.
        :type time_dependent_op: str or Callable or None
        :param timeless_op: An operation to be used to join data for any 
            timeless raster feature. Supported options are:
            - None (default): If arrays are the same, take one. 
                Raise an error otherwise.
            - 'concatenate': Join arrays over the last (i.e. bands) dimension
            - 'min': Join arrays by taking minimum values. Ignore NaN values.
            - 'max': Join arrays by taking maximum values. Ignore NaN values.
            - 'mean': Join arrays by taking mean values. Ignore NaN values.
            - 'median': Join arrays by taking median values. Ignore NaN values.
        :type timeless_op: str or Callable or None
        :return: A dictionary with OOI data and values
        :rtype: Dict[(DataType, str), object]
        """
        ooi_content = merge_oois(self, *oois, data=data,
                                          time_dependent_op=time_dependent_op, 
                                          timeless_op=timeless_op)

        merged_ooi = OOI()
        for data, value in ooi_content.items():
            merged_ooi[data] = value
        return merged_ooi

    def time_series(self, ref_date=None, scale_time=1):
        """
        Returns a numpy array with seconds passed between the reference date 
        and the timestamp of each image.

        An array is constructed as time_series[i] = 
                                (timestamp[i] - ref_date).total_seconds().
        If reference date is None the first date in the OOI's timestamp is taken.
        If OOI timestamp attribute is empty the method returns None.
        
        :param ref_date: reference date relative to which the time is measured
        :type ref_date: datetime object
        :param scale_time: scale seconds by factor. 
            If `60`, time will be in minutes, if `3600` hours
        :type scale_time: int
        """

        if not self.timestamp:
            return None

        if ref_date is None:
            ref_date = self.timestamp[0]

        return np.asarray([round((timestamp - ref_date).total_seconds()\
                                 / scale_time) for timestamp in self.timestamp],
                                  dtype=np.int64)

    def consolidate_timestamps(self, timestamps):
        """
        Removes all frames from the OOI with a date not found in the provided 
            timestamps list.
        
        :param timestamps: keep frames with date found in this list
        :type timestamps: list of datetime objects
        :return: set of removed frames' dates
        :rtype: set of datetime objects
        """
        remove_from_patch = set(self.timestamp).difference(timestamps)
        remove_from_patch_idxs = [self.timestamp.index(rm_date)\
                                  for rm_date in remove_from_patch]
        good_timestamp_idxs = [idx for idx, _ in enumerate(self.timestamp)\
                               if idx not in remove_from_patch_idxs]
        good_timestamps = [date for idx, date in enumerate(self.timestamp)\
                           if idx not in remove_from_patch_idxs]

        for data_type in [data_type for data_type in DataType if\
                          (data_type.is_time_dependent() and data_type.has_dict())]:

            for ooi_name, value in self[data_type].items():
                if isinstance(value, np.ndarray):
                    self[data_type][ooi_name] = value[good_timestamp_idxs, ...]
                if isinstance(value, list):
                    self[data_type][ooi_name] = [value[idx]\
                                                 for idx in good_timestamp_idxs]

        self.timestamp = good_timestamps
        return remove_from_patch

    def plot(self, data, 
             rgb=None, rgb_factor=3.5, vdims=None, 
             timestamp_column='TIMESTAMP',
             geometry_column='geometry', 
             pixel=False, mask=None):
        """ 
        Plots ooi data

        :param data: data of ooi
        :type data: (DataType, str)
        :param rgb: indexes of bands to create rgb image from
        :type rgb: [int, int, int]
        :param rgb_factor: factor for rgb bands multiplication
        :type rgb_factor: float
        :param vdims: value dimension for vector data
        :type vdims: str
        :param timestamp_column: name of the timestamp column, valid for vector data
        :type timestamp_column: str
        :param geometry_column: name of the geometry column, valid for vector data
        :type geometry_column: str
        :param pixel: plot values through time for one pixel
        :type pixel: bool
        :param mask: where eopatch[FeatureType.MASK] == False, value = 0
        :type mask: str
        :return: plot
        :rtype: holovies/bokeh
        """

        ## TODO!
        # from eolearn.visualization import EOPatchVisualization
#        vis = EOPatchVisualization(self, feature=feature, rgb=rgb, rgb_factor=rgb_factor, vdims=vdims,
#                                   timestamp_column=timestamp_column, geometry_column=geometry_column,
#                                   pixel=pixel, mask=mask)
#        return vis.plot()

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

class _DataDict(dict):
    """
    A dictionary structure that holds data of certain data type.
    It checks that data have a correct dimension. 
    It also supports lazy loading by accepting a function as a data value, 
    which is then called when the data is accessed.
    """
    FORBIDDEN_CHARS = {'.', '/', '\\', '|', ';', ':', '\n', '\t'}

    def __init__(self, data_dict, data_type):
        """
        :param data_dict: A dictionary of data names and values
        :type data_dict: dict(str: object)
        :param data_type: Type of data
        :type data_type: DataType
        """
        super().__init__()

        self.data_type = data_type
        self.ndim = self.data_type.ndim()
        self.is_vector = self.data_type.is_vector()

        for ooi_name, value in data_dict.items():
            self[ooi_name] = value

    def __setitem__(self, ooi_name, value):
        """ 
        Before setting value to the dictionary it checks that value is of 
        correct type and dimension and tries to transform value in correct form.
        """
        value = self._parse_data_value(value)
        self._check_data_name(ooi_name)
        super().__setitem__(ooi_name, value)

    def _check_data_name(self, ooi_name):
        if not isinstance(ooi_name, str):
            error_msg = "Feature name must be a string but an object of\
            type {} was given."
            raise ValueError(error_msg.format(type(ooi_name)))

        for char in ooi_name:
            if char in self.FORBIDDEN_CHARS:
                error_msg = "The name of feature ({}, {}) contains an\
                    illegal character '{}'."
                raise ValueError(error_msg.format(self.data_type, 
                                                  ooi_name, char))

        if ooi_name == '':
            raise ValueError("Feature name cannot be an empty string.")

    def __getitem__(self, ooi_name, load=True):
        """Implements lazy loading."""
        value = super().__getitem__(ooi_name)

        if isinstance(value, DataIO) and load:
           value = value.load()
           self[ooi_name] = value

        return value

    def get_dict(self):
        """
        Returns a Python dictionary of data and value.
        """
        return dict(self)

    def _parse_data_value(self, value):
        """ 
        Checks if value fits the data type. 
        If not it tries to fix it or raise an error
        
        :raises: ValueError
        """
        if isinstance(value, DataIO):
           return value
        if not hasattr(self, 'ndim'):  # Because of serialization/deserialization during multiprocessing
            return value

        if self.ndim:
            if not isinstance(value, np.ndarray):
                raise ValueError('{} data has to be a numpy array'.\
                                 format(self.data_type))
            if value.ndim != self.ndim:
                raise ValueError('Numpy array of {} data has to have {} '
                                 'dimension{}'.format(self.data_type, 
                                                      self.ndim, 's'\
                                                          if self.ndim > 1 else ''))

            if self.data_type.is_discrete():
                if not issubclass(value.dtype.type, 
                                  (np.integer, bool, np.bool_, np.bool8)):
                    msg = '{} is a discrete data type therefore dtype of\
                        data should be a subtype of ' \
                          'numpy.integer or numpy.bool, found type {}.\
                              In the future an error will be raised because ' \
                                  'of this'.\
                                      format(self.data_type, value.dtype.type)
                    warnings.warn(msg, DeprecationWarning, stacklevel=3)

            # This checking is disabled for now
            # else:
            #     if not issubclass(value.dtype.type, (np.floating, np.float)):
            #         raise ValueError('{} is a floating feature type therefore dtype of data has to be a subtype of '
            #                          'numpy.floating or numpy.float, found type {}'.format(self.feature_type,
            #                                                                                value.dtype.type))
            return value

        if self.is_vector:
            if isinstance(value, gpd.GeoSeries):
                value = gpd.GeoDataFrame(dict(geometry=value), crs=value.crs)

            if isinstance(value, gpd.GeoDataFrame):
                if self.data_type is DataType.VECTOR:
                    if DataType.TIMESTAMP.value.upper() not in value:
                        raise ValueError("{} data has to contain a column\
                                         'TIMESTAMP' with timestamps".\
                                             format(self.data_type))

                return value

            raise ValueError('{} data works with data of type {},\
                             parsing data type {} is not supported given'.\
                                 format(self.data_type, 
                                        gpd.GeoDataFrame.__name__, type(value)))

        return value

class DataParser:
    """ 
    Takes a collection of data structured in a various ways and 
    parses them into one way. 
    It can parse data straight away or it can parse them only if they 
    exist in a given `OOI`. 
    If input format is not recognized or data don't exist in a given `OOI` 
    it raises an error. The class is a generator therefore parsed data
    can be obtained by iterating over an instance of the class. 
    An `OOI` is given as a parameter of the generator.
    
    Supported input formats:
    - Anything that exists in a given `OOI` is defined with `...`
    - A data type describing all data of that type. 
            Example: `DataType.DATA` or `DataType.BBOX`
    - Single data as a tuple. 
            Example: `(DataType.DATA, 'BANDS')`
    - Single data as a tuple. 
            Example: `(DataType.DATA, 'BANDS')`
    - Single data as a tuple with new name. 
            Example `(DataType.DATA, 'BANDS', 'NEW_BANDS')`
    - A list of data (new names or not).
            Example:
                [
                (DataType.DATA, 'BANDS'),
                (DataType.MASK, 'CLOUD_MASK', 'NEW_CLOUD_MASK')
                ]

    - A dictionary with data types as keys and lists, sets, single data 
        or `...` of data names as values.
            Example:
                {
                DataType.DATA: ['S2-BANDS', 'L8-BANDS'],
                DataType.MASK: {'IS_VALID', 'IS_DATA'},
                DataType.MASK_TIMELESS: 'LULC',
                DataType.TIMESTAMP: ...
                }
     - A dictionary with data types as keys and dictionaries, where data names 
     are mapped into new names, as values. 
             Example:
                 {
                DataType.DATA: {
                    'S2-BANDS': 'INTERPOLATED_S2_BANDS',
                    'L8-BANDS': 'INTERPOLATED_L8_BANDS',
                    'NDVI': ...
                }
                }
    Outputs of the generator:
    - tuples in form of (data type, ooi name) if parameter `new_names=False`
    - tuples in form of (data type, ooi name, new feature name) 
        if parameter `new_names=True`
    """

    def __init__(self, data, new_names=False, 
                 rename_function=None, 
                 default_data_type=None,
                 allowed_data_types=None):
        """
        :param data: A collection of data in one of the supported formats
        :type data: object
        :param new_names: If `False` the generator will only return tuples 
            with in form of
            (data type, ooi name). 
            If `True` it will return tuples
            (data type, ooi name, new ooi name) which can be used for renaming
            data or creating new data out of old ones.
        :type new_names: bool
        :param rename_function: A function which transforms ooi name into a 
            new ooi name, default is identity function. 
            This parameter is only applied if `new_names` is set to `True`.
        :type rename_function: function or None
        :param default_data_type: If data type of any given data is not set, 
            this will be used. 
            By default this is set to `None`. In this case if data type of 
                any data is not given the following will happen:
            - if iterated over `OOI` - It will try to find data with 
                matching name in OOI. If such data exist, it will return any 
                of them. Otherwise it will raise an error.
            - if iterated without `OOI` - It will return `...` instead 
                of a data type.
        :type default_data_type: DataType or None
        :param allowed_data_types: Makes sure that only data of these 
            data types will be returned, otherwise an error is raised
        :type: set(DataType) or None
        :raises: ValueError
        
        """
        self.data_collection = self._parse_data(data, new_names)
        self.new_names = new_names
        self.rename_function = rename_function
        self.default_data_type = default_data_type
        self.allowed_data_types = DataType\
            if allowed_data_types is None else set(allowed_data_types)

        if rename_function is None:
            self.rename_function = self._identity_rename_function  # <- didn't use lambda function - it can't be pickled

        if allowed_data_types is not None:
            self._check_data_types()

    def __call__(self, ooi=None):
        return self._get_data(ooi)

    def __iter__(self):
        return self._get_data()

    @staticmethod
    def _parse_data(data, new_names):
        """
        Takes a collection of data structured in a various ways and 
        parses them into one way.
        If input format is not recognized it raises an error.
        
        :return: A collection of data
        :rtype: collections.OrderedDict(DataType: collections.OrderedDict(str: str or Ellipsis) or Ellipsis)
        :raises: ValueError
        """
        if isinstance(data, dict):
            return DataParser._parse_dict(data, new_names)

        if isinstance(data, list):
            return DataParser._parse_list(data, new_names)

        if isinstance(data, tuple):
            return DataParser._parse_tuple(data, new_names)

        if data is ...:
            return OrderedDict([(data_type, ...) for data_type in DataType])

        if isinstance(data, DataType):
            return OrderedDict([(data, ...)])

        if isinstance(data, str):
            return OrderedDict([(None, OrderedDict([(data, ...)]))])

        raise ValueError('Unknown format of input data: {}'.format(data))

    @staticmethod
    def _parse_dict(data, new_names):
        """Helping function of `_parse_data` that parses a list."""
        data_collection = OrderedDict()
        for data_type, ooi_names in data.items():
            try:
                data_type = DataType(data_type)
            except ValueError:
                ValueError('Failed to parse {}, keys of the dictionary have to be instances '
                           'of {}'.format(data, DataType.__name__))

            data_collection[data_type] = data_collection.get(data_type, OrderedDict())

            if ooi_names is ...:
                data_collection[data_type] = ...

            if data_type.has_dict() and data_collection[data_type] is not ...:
                data_collection[data_type].\
                    update(DataParser._parse_data_names(ooi_names, new_names))

        return data_collection

    @staticmethod
    def _parse_list(data, new_names):
        """Helping function of `_parse_features` that parses a list."""
        data_collection = OrderedDict()
        for dat in data:
            if isinstance(dat, DataType):
                data_collection[dat] = ...

            elif isinstance(dat, (tuple, list)):
                for data_type, data_dict\
                    in DataParser._parse_tuple(dat, new_names).items():
                    data_collection[data_type] =\
                        data_collection.get(data_type, OrderedDict())

                    if data_dict is ...:
                        data_collection[data_type] = ...

                    if data_collection[data_type] is not ...:
                        data_collection[data_type].update(data_dict)
            else:
                raise ValueError('Failed to parse {}, expected a tuple'.\
                                 format(dat))
        return data_collection

    @staticmethod
    def _parse_tuple(data, new_names):
        """Helping function of `_parse_features` that parses a tuple."""
        name_idx = 1
        try:
            data_type = DataType(data[0])
        except ValueError:
            data_type = None
            name_idx = 0

        if data_type and not data_type.has_dict():
            return OrderedDict([(data_type, ...)])
        return OrderedDict([(data_type, 
                             DataParser._parse_names_tuple(data[name_idx:], 
                                                           new_names))])
    @staticmethod
    def _parse_data_names(ooi_names, new_names):
        """Helping function of `_parse_data` that parses a collection of data names."""
        if isinstance(ooi_names, set):
            return DataParser._parse_names_set(ooi_names)

        if isinstance(ooi_names, dict):
            return DataParser._parse_names_dict(ooi_names)

        if isinstance(ooi_names, (tuple, list)):
            return DataParser._parse_names_tuple(ooi_names, new_names)

        raise ValueError('Failed to parse {}, expected dictionary,\
                         set or tuple'.format(ooi_names))

    @staticmethod
    def _parse_names_set(ooi_names):
        """Helping function of `_parse_feature_names` that parses a set of feature names."""
        data_collection = OrderedDict()
        for ooi_name in ooi_names:
            if isinstance(ooi_name, str):
                data_collection[ooi_name] = ...
            else:
                raise ValueError('Failed to parse {}, expected string'.\
                                 format(ooi_name))
        return data_collection

    @staticmethod
    def _parse_names_dict(ooi_names):
        """Helping function of `_parse_feature_names` that parses a dictionary of feature names."""
        data_collection = OrderedDict()
        for ooi_name, new_ooi_name in ooi_names.items():
            if isinstance(ooi_name, str) and (isinstance(new_ooi_name, str) or
                                                  new_ooi_name is ...):
                data_collection[ooi_name] = new_ooi_name
            else:
                if not isinstance(ooi_name, str):
                    raise ValueError('Failed to parse {},\
                                     expected string'.format(ooi_name))
                raise ValueError('Failed to parse {}, expected\
                                 string or Ellipsis'.format(new_ooi_name))
        return data_collection

    @staticmethod
    def _parse_names_tuple(ooi_names, new_names):
        """Helping function of `_parse_feature_names` that parses a tuple or a 
        list of data names."""
        for name in ooi_names:
            if not isinstance(name, str) and name is not ...:
                raise ValueError('Failed to parse {}, expected a string'.format(name))

        if ooi_names[0] is ...:
            return ...

        if new_names:
            if len(ooi_names) == 1:
                return OrderedDict([(ooi_names[0], ...)])
            if len(ooi_names) == 2:
                return OrderedDict([(ooi_names[0], ooi_names[1])])
            raise ValueError("Failed to parse {}, it should contain at\
                             most two strings".format(ooi_names))

        if ... in ooi_names:
            return ...
        return OrderedDict([(name, ...) for name in ooi_names])
    
    def _check_data_types(self):
        """ 
        Checks that data types are a subset of allowed data types. 
        (`None` is handled
        :raises: ValueError
        """
        if self.default_data_type is not None and\
            self.default_data_type not in self.allowed_data_types:
            raise ValueError('Default data type parameter must be one\
                             of the allowed data types')

        for data_type in self.data_collection:
            if data_type is not None and data_type not in self.allowed_data_types:
                raise ValueError('Data type has to be one of {}, but {} found'.\
                                 format(self.allowed_data_types, data_type))

    def _get_data(self, ooi=None):

        """A generator of parsed data.
        :param ooi: A given OOI
        :type ooih: OOI or None
        :return: One by one data
        :rtype: tuple(DataType, str) or tuple(DataType, str, str)
        """
        for data_type, data_dict in self.data_collection.items():
            if data_type is None and self.default_data_type is not None:
                data_type = self.default_data_type

            if data_type is None:
                for ooi_name, new_ooi_name in data_dict.items():
                    if ooi is None:
                        yield self._return_data(..., ooi_name, new_ooi_name)
                    else:
                        found_data_type = self._find_feature_type(ooi_name, ooi)
                        if found_data_type:
                            yield self._return_data(found_data_type, 
                                                    ooi_name, new_ooi_name)
                        else:
                            raise ValueError("Data with name '{}' does not\
                                             exist among data of allowed data"
                                             " types in given OOI.\
                                                 Allowed datatypes are "
                                             "{}".format(ooi_name, self.allowed_data_types))

            elif data_dict is ...:
                if not data_type.has_dict() or ooi is None:
                    yield self._return_data(data_type, ...)
                else:
                    for ooi_name in ooi[data_type]:
                        yield self._return_data(data_type, ooi_name)
            else:
                for ooi_name, new_ooi_name in data_dict.items():
                    if ooi is not None and ooi_name not in ooi[data_type]:
                        raise ValueError('Data {} of type {} was not found in OOI'.\
                                         format(ooi_name, data_type))
                    yield self._return_data(data_type, ooi_name, new_ooi_name)

    def _find_data_type(self, ooi_name, ooi):
        """ 
        Iterates over allowed data types of given OOI and tries to find a 
        data type for which there exists data with given name
        :return: A data type or `None` if such data type does not exist
        :rtype: DataType or None
        """
        for data_type in self.allowed_data_types:
            if data_type.has_dict() and ooi_name in ooi[data_type]:
                return data_type
        return None

    def _return_data(self, data_type, ooi_name, new_ooi_name=...):
        """ 
        Helping function of `get_data`
        """
        if self.new_names:
            return data_type, ooi_name, (self.rename_function(ooi_name)\
                                         if new_ooi_name is ... else
                                                new_ooi_name)
        return data_type, ooi_name

    @staticmethod
    def _identity_rename_function(name):
        return name

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

    for ftype, fname in DataParser(data):
        if fname is ... and not existing_data[ftype]:
            continue

        if ftype.is_meta():
            if ftype in returned_meta_data:
                continue
            fname = ...
            returned_meta_data.add(ftype)

        elif ftype not in queried_data and (fname is ...\
                                            or fname not in existing_data[ftype]):
            queried_data.add(ftype)
            if ... not in existing_data[ftype]:
                raise IOError('There are no data of type {} in saved OOI'.\
                              format(ftype))

            for ooi_name, path in walk_data_type_folder(filesystem, 
                    existing_data[ftype][...]):
                existing_data[ftype][ooi_name] = path

        if fname not in existing_data[ftype]:
            raise IOError('Data {} does not exist in saved OOI'.\
                          format((ftype, fname)))

        if fname is ... and not ftype.is_meta():
            for ooi_name, path in existing_data[ftype].items():
                if ooi_name is not ...:
                    yield ftype, ooi_name, path
        else:
            yield ftype, fname, existing_data[ftype][fname]

def walk_ooi(ooi, patch_location, data=...):
    """ 
    Recursively reads a patch_location and returns yields tuples of 
    (data_type, ooi_name, file_path)
    """
    returned_meta_data = set()
    for ftype, fname in DataParser(data)(ooi):
        name_basis = fs.path.combine(patch_location, ftype.value)
        if ftype.is_meta():
            if ooi[ftype] and ftype not in returned_meta_data:
                yield ftype, ..., name_basis
                returned_meta_data.add(ftype)
        else:
            yield ftype, fname, fs.path.combine(name_basis, fname)

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


def save_ooi(ooi, filesystem, patch_location, data=..., 
             overwrite_permission=OverwritePermission.ADD_ONLY, compress_level=0):
    """ 
    A utility function used by OOI.save method
    """

    data_exists = filesystem.exists(patch_location)
    if overwrite_permission is OverwritePermission.OVERWRITE_DATA and data_exists:
        filesystem.removetree(patch_location)
        if patch_location != '/':
            data_exists = False

    if not data_exists:
        filesystem.makedirs(patch_location, recreate=True)

    ooi_data = list(walk_ooi(ooi, patch_location, data))

    if overwrite_permission is OverwritePermission.ADD_ONLY or \
        (sys_is_windows()\
         and overwrite_permission is OverwritePermission.OVERWRITE_FEATURES):
        fs_data = list(walk_filesystem(filesystem, patch_location))
    else:
        fs_data = []

    _check_case_matching(ooi_data, fs_data)

    if overwrite_permission is OverwritePermission.ADD_ONLY:
        _check_add_only_permission(ooi_data, fs_data)

    ftype_folder_map = {(ftype, fs.path.dirname(path))\
                        for ftype, _, path in ooi_data if not ftype.is_meta()}

    for ftype, folder in ftype_folder_map:
        if not filesystem.exists(folder):
            filesystem.makedirs(folder, recreate=True)

    data_to_save = ((DataIO(filesystem, path),
                         ooi[(ftype, fname)],
                         DataFormat.NPY if ftype.is_raster() else DataFormat.PICKLE,
                         compress_level) for ftype, fname, path in ooi_data)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # The following is intentionally wrapped in a list in order to get 
        # back potential exceptions
        list(executor.map(lambda params: params[0].save(*params[1:]), data_to_save))

def load_ooi(ooi, filesystem, patch_location, data=..., lazy_loading=False):
    """ 
    A utility function used by OOI.load method
    """
    data_list = list(walk_filesystem(filesystem, patch_location, data))
    loading_data = [DataIO(filesystem, path) for _, _, path in data_list]

    if not lazy_loading:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            loading_data = executor.map(lambda loader: loader.load(), loading_data)

    for (ftype, fname, _), value in zip(data_list, loading_data):
        ooi[(ftype, fname)] = value
    
    return ooi

