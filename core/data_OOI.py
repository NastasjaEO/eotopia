# -*- coding: utf-8 -*-
"""
Created on Thu May  6 19:38:36 2021

@author: nasta
"""

import attr
import copy
import logging
import warnings

import numpy as np
import geopandas as gpd

import dateutil.parser
import datetime

from sentinelhub import BBox, CRS

## TODO!
#from .data_types import DataType, OverwritePermission
#from .data_OOI_IO import save_ooi, load_ooi, DataIO
#from .utils.filesystem_utils import get_filesystem

import sys
sys.path.append("D:/Code/eotopia/core")
from data_OOI_IO import save_ooi, load_ooi, DataIO
from data_types import DataType, OverwritePermission
from data_OOI_utils import DataParser

sys.path.append("D:/Code/eotopia/utils")
from filesystem_utils import get_filesystem

LOGGER = logging.getLogger(__name__)
warnings.simplefilter('default', DeprecationWarning)

MAX_DATA_REPR_LEN = 100


@attr.s(repr=False, eq=False, kw_only=True)
class OOI:
    """
    Object Of Interest
    Usually an EO data thingy
    """

    data = attr.ib(factory=dict)
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
            ## TODO!
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

    ## TODO!
    # def reset_feature_type(self, data_type):

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

    def merge(self, *ooi, data=..., 
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
        ## TODO!
#        ooi_content = merge_eopatches(self, *oois, data=data,
#                                          time_dependent_op=time_dependent_op, 
#                                          timeless_op=timeless_op)

        merged_ooi = OOI()
#        for data, value in ooi_content.items():
#            merged_ooi[data] = value

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
#        try:
#            from eolearn.visualization import EOPatchVisualization
#        except ImportError:
#            raise RuntimeError('Subpackage eo-learn-visualization has to be installed with an option [FULL] in order '
#                               'to use plot method')

#        vis = EOPatchVisualization(self, feature=feature, rgb=rgb, rgb_factor=rgb_factor, vdims=vdims,
#                                   timestamp_column=timestamp_column, geometry_column=geometry_column,
#                                   pixel=pixel, mask=mask)
#        return vis.plot()
        
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
