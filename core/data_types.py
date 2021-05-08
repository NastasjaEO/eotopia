# -*- coding: utf-8 -*-
"""
Created on Fri May  7 12:42:43 2021

@author: nasta
"""

from enum import Enum
from sentinelhub import BBox


class DataType(Enum):
    """
    The Enum class of all possible feature types that can be included in
    an analysis
    """

    DATA = 'data'
    MASK = 'mask'
    SCALAR = 'scalar'
    LABEL = 'label'
    VECTOR = 'vector'
    DATA_TIMELESS = 'data_timeless'
    MASK_TIMELESS = 'mask_timeless'
    SCALAR_TIMELESS = 'scalar_timeless'
    LABEL_TIMELESS = 'label_timeless'
    VECTOR_TIMELESS = 'vector_timeless'
    META_INFO = 'meta_info'
    BBOX = 'bbox'
    TIMESTAMP = 'timestamp'

    @classmethod
    def has_value(cls, value):
        """True if value is in DataType values. False otherwise."""
        return value in cls._value2member_map_

    def is_spatial(self):
        """True if DataType has a spatial component. False otherwise."""
        return self in DataTypeSet.SPATIAL_TYPES

    def is_time_dependent(self):
        """True if DataType has a time component. False otherwise."""
        return self in DataTypeSet.TIME_DEPENDENT_TYPES

    def is_timeless(self):
        """True if DataType doesn't have a time component and is not a 
        meta feature. False otherwise."""
        return self in DataTypeSet.TIMELESS_TYPES

    def is_discrete(self):
        """True if DataType should have discrete (integer) values. 
        False otherwise."""
        return self in DataTypeSet.DISCRETE_TYPES

    def is_meta(self):
        """ True if DataType is for storing metadata info and False otherwise. """
        return self in DataTypeSet.META_TYPES

    def is_vector(self):
        """True if DataType is vector feature type. False otherwise. """
        return self in DataTypeSet.VECTOR_TYPES

    def has_dict(self):
        """True if DataType stores a dictionary. False otherwise."""
        return self in DataTypeSet.DICT_TYPES

    def is_raster(self):
        """True if DataType stores a dictionary with raster data. 
        False otherwise."""
        return self in DataTypeSet.RASTER_TYPES

    def contains_ndarrays(self):
        """True if DataType stores a dictionary of numpy.ndarrays. 
        False otherwise."""
        return self in DataTypeSet.RASTER_TYPES

    def ndim(self):
        """If given DataType stores a dictionary of numpy.ndarrays it 
        returns dimensions of such arrays."""
        if self.is_raster():
            return {
                DataType.DATA: 4,
                DataType.MASK: 4,
                DataType.SCALAR: 2,
                DataType.LABEL: 2,
                DataType.DATA_TIMELESS: 3,
                DataType.MASK_TIMELESS: 3,
                DataType.SCALAR_TIMELESS: 1,
                DataType.LABEL_TIMELESS: 1
            }[self]
        return None

    def type(self):
        """Returns type of the data for the given DataType."""
        if self is DataType.TIMESTAMP:
            return list
        if self is DataType.BBOX:
            return BBox
        return dict

class DataTypeSet:
    """ 
    A collection of immutable sets of data types, grouped together by 
    certain properties.
    """
    SPATIAL_TYPES = frozenset([DataType.DATA, 
                               DataType.MASK, 
                               DataType.VECTOR, 
                               DataType.DATA_TIMELESS,
                               DataType.MASK_TIMELESS, 
                               DataType.VECTOR_TIMELESS])
    TIME_DEPENDENT_TYPES = frozenset([DataType.DATA, 
                                      DataType.MASK, 
                                      DataType.SCALAR, 
                                      DataType.LABEL,
                                      DataType.VECTOR, 
                                      DataType.TIMESTAMP])
    TIMELESS_TYPES = frozenset([DataType.DATA_TIMELESS, 
                                DataType.MASK_TIMELESS, 
                                DataType.SCALAR_TIMELESS,
                                DataType.LABEL_TIMELESS, 
                                DataType.VECTOR_TIMELESS])
    DISCRETE_TYPES = frozenset([DataType.MASK, 
                                DataType.MASK_TIMELESS, 
                                DataType.LABEL,
                                DataType.LABEL_TIMELESS])
    META_TYPES = frozenset([DataType.META_INFO, 
                            DataType.BBOX, 
                            DataType.TIMESTAMP])
    VECTOR_TYPES = frozenset([DataType.VECTOR, 
                              DataType.VECTOR_TIMELESS])
    RASTER_TYPES = frozenset([DataType.DATA, 
                              DataType.MASK, 
                              DataType.SCALAR, 
                              DataType.LABEL,
                              DataType.DATA_TIMELESS, 
                              DataType.MASK_TIMELESS, 
                              DataType.SCALAR_TIMELESS,
                              DataType.LABEL_TIMELESS])
    DICT_TYPES = frozenset([DataType.DATA, 
                            DataType.MASK, 
                            DataType.SCALAR, 
                            DataType.LABEL,
                            DataType.VECTOR, 
                            DataType.DATA_TIMELESS, 
                            DataType.MASK_TIMELESS,
                            DataType.SCALAR_TIMELESS, 
                            DataType.LABEL_TIMELESS, 
                            DataType.VECTOR_TIMELESS,
                            DataType.META_INFO])
    RASTER_TYPES_4D = frozenset([DataType.DATA, 
                                 DataType.MASK])
    RASTER_TYPES_3D = frozenset([DataType.DATA_TIMELESS, 
                                 DataType.MASK_TIMELESS])
    RASTER_TYPES_2D = frozenset([DataType.SCALAR, 
                                 DataType.LABEL])
    RASTER_TYPES_1D = frozenset([DataType.SCALAR_TIMELESS, 
                                 DataType.LABEL_TIMELESS])


class DataFormat(Enum):
    """ Enum class for file formats used for saving and loading Data
    """

    PICKLE = 'pkl'
    NPY = 'npy'
    GZIP = 'gz'

    def extension(self):
        """ Returns file extension of file format
        """
        return '.{}'.format(self.value)

    @staticmethod
    def split_by_extensions(filename):
        """ Splits the filename string by the extension of the file
        """
        parts = filename.split('.')
        idx = len(parts) - 1
        while DataFormat().is_file_format(parts[idx]):
            parts[idx] = DataFormat(parts[idx])
            idx -= 1
        return ['.'.join(parts[:idx + 1])] + parts[idx + 1:]

    @classmethod
    def is_file_format(cls, value):
        """ Tests whether value represents one of the supported file formats
        :param value: The string representation of the enum constant
        :type value: str
        :return: `True` if string is file format and `False` otherwise
        :rtype: bool
        """
        return any(value == item.value for item in cls)

class OverwritePermission(Enum):
    """ 
    Enum class which specifies which content of saved Data can be overwritten 
    when saving new content.
    Permissions are in the following hierarchy:
    - `ADD_ONLY` - Only new features can be added, anything that is already 
        saved cannot be changed.
    - `OVERWRITE_FEATURES` - Overwrite only data for features which have to be 
        saved. The remaining content of saved
        Data will stay unchanged.
    - `OVERWRITE_DATA` - Overwrite entire content of saved Data and replace it 
        with the new content.
    """
    ADD_ONLY = 0
    OVERWRITE_FEATURES = 1
    OVERWRITE_DATA = 2




