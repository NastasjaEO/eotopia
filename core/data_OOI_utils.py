# -*- coding: utf-8 -*-
"""
Created on Fri May  7 22:23:59 2021

@author: nasta
"""

from collections import OrderedDict

import numpy as np
import geopandas as gpd
from geopandas.testing import assert_geodataframe_equal

import sys
sys.path.append("D:/Code/eotopia/core")
from data_types import DataType, OverwritePermission, DataFormat


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
        """Helping function of `_parse_feature_names` that parses a tuple or a list of feature names."""
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
                        yield self._return_feature(data_type, ooi_name)
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
        """ Helping function of `get_data`
        """
        if self.new_names:
            return data_type, ooi_name, (self.rename_function(ooi_name)\
                                         if new_ooi_name is ... else
                                                new_ooi_name)
        return data_type, ooi_name

    @staticmethod
    def _identity_rename_function(name):
        return name

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


