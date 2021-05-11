# -*- coding: utf-8 -*-
"""
Created on Tue May 11 10:25:22 2021

@author: freeridingeo
"""

import copy
import warnings
from abc import abstractmethod

import fs
import numpy as np

from .eodata import OOI
from .OOITask import OOITask
from .fs_utils import get_filesystem

import sys
sys.path.append("D:/Code/eotopia/core")
from data_OOI_classes import OOI
from OOI_task_classes import OOITask
sys.path.append("D:/Code/eotopia/utils")
from filesystem_utils import (get_filesystem, 
                              walk_data_type_folder, walk_main_folder,
                              _check_add_only_permission, _check_case_matching)

warnings.simplefilter('default', DeprecationWarning)


class CopyTask(OOITask):
    """
    Makes a shallow copy of the given OOI.
    It copies data type dictionaries but not the data itself.
    """
    def __init__(self, data=...):
        """
        :param data: A collection of data or data types that will be copied 
            into a new OOI.
        :type data: an object supported by the DataParser class
        """
        self.data = data

    def execute(self, OOI):
        return OOI.__copy__(data=self.data)

class DeepCopyTask(CopyTask):
    """ 
    Makes a deep copy of the given OOI.
    """
    def execute(self, OOI):
        return OOI.__deepcopy__(data=self.data)

class IOTask(OOITask):
    """ 
    An abstract Input/Output task that can handle a path and a filesystem object
    """
    def __init__(self, path, filesystem=None, create=False, config=None):
        """
        :param path: root path where all OOIes are saved
        :type path: str
        :param filesystem: An existing filesystem object. 
            If not given it will be initialized according to the OOI path. 
            If you intend to run this task in multiprocessing mode you shouldn't 
                specify this parameter.
        :type filesystem: fs.base.FS or None
        :param create: If the filesystem path doesn't exist this flag indicates 
            to either create it or raise an error
        :type create: bool
        :param config: A configuration object with AWS credentials. 
            By default is set to None and in this case the default configuration 
                will be taken.
        :type config: SHConfig or None
        """
        self.path = path
        self._filesystem = filesystem
        self._create = create
        self.config = config
        self.filesystem_path = '/' if self._filesystem is None else self.path

    @property
    def filesystem(self):
        """ 
        A filesystem property that is being lazy-loaded the first time it is needed
        """
        if self._filesystem is None:
            self._filesystem = get_filesystem(self.path, create=self._create, 
                                              config=self.config)
        return self._filesystem

    @abstractmethod
    def execute(self, *OOIes, **kwargs):
        """ 
        Implement execute function
        """
        raise NotImplementedError

class SaveTask(IOTask):
    """ 
    Saves the given OOI to a filesystem
    """
    def __init__(self, path, filesystem=None, config=None, **kwargs):
        """
        :param path: root path where all OOIes are saved
        :type path: str
        :param filesystem: An existing filesystem object. 
            If not given it will be initialized according to the OOI path. 
            If you intend to run this task in multiprocessing mode you shouldn't 
                specify this parameter.
        :type filesystem: fs.base.FS or None
        :param data: A collection of data types specifying data of which 
            type will be saved. By default all data will be saved.
        :type data: an object supported by the DataParser
        :param overwrite_permission: A level of permission for overwriting an existing OOI
        :type overwrite_permission: OverwritePermission or int
        :param compress_level: A level of data compression and can be specified 
            with an integer from 0 (no compression) to 9 (highest compression).
        :type compress_level: int
        :param config: A configuration object with AWS credentials. 
            By default is set to None and in this case the default configuration 
            will be taken.
        :type config: SHConfig or None
        """
        self.kwargs = kwargs
        super().__init__(path, filesystem=filesystem, create=True, config=config)

    def execute(self, OOI, *, OOI_folder=''):
        """
        Saves the OOI to disk: `folder/OOI_folder`.
        
        :param OOI: OOI which will be saved
        :type OOI: OOI
        :param OOI_folder: name of OOI folder containing data
        :type OOI_folder: str
        :return: The same OOI
        :rtype: OOI
        """
        path = fs.path.combine(self.filesystem_path, OOI_folder)

        OOI.save(path, filesystem=self.filesystem, **self.kwargs)
        return OOI

class LoadTask(IOTask):
    """ 
    Loads an OOI from a filesystem
    """
    def __init__(self, path, filesystem=None, config=None, **kwargs):
        """
        :param path: root directory where all OOIes are saved
        :type path: str
        :param filesystem: An existing filesystem object. 
            If not given it will be initialized according to the OOI path. 
            If you intend to run this task in multiprocessing mode you shouldn't 
            specify this parameter.
        :type filesystem: fs.base.FS or None
        :param data: A collection of data to be loaded. By default all datawill be loaded.
        :type data: an object supported by the DataParser class
        :param lazy_loading: If `True` data will be lazy loaded. Default is `False`
        :type lazy_loading: bool
        :param config: A configuration object with AWS credentials. 
            By default is set to None and in this case the default configuration 
            will be taken.
        :type config: SHConfig or None
        """
        self.kwargs = kwargs
        super().__init__(path, filesystem=filesystem, create=False, config=config)

    def execute(self, *, OOI_folder=''):
        """
        Loads the OOI from disk: `folder/OOI_folder`.
        
        :param OOI_folder: name of OOI folder containing data
        :type OOI_folder: str
        :return: OOI loaded from disk
        :rtype: OOI
        """
        path = fs.path.combine(self.filesystem_path, OOI_folder)

        return OOI.load(path, filesystem=self.filesystem, **self.kwargs)

class AddData(OOITask):
    """Adds data to the given OOI.
    """
    def __init__(self, data):
        """
        :param data: Data to be added
        :type data: (DataType, data_name) or DataType
        """
        self.data_type, self.data_name = next(self._parse_data(data)())

    def execute(self, OOI, new_data):
        """Returns the OOI with added features.
        :param OOI: input OOI
        :type OOI: OOI
        :param new_data: data to be added to the data
        :type new_data: object
        :return: input OOI with the specified data
        :rtype: OOI
        """
        if self.data_name is None:
            OOI[self.data_type] = new_data
        else:
            OOI[self.data_type][self.data_name] = new_data
        return OOI

class RemoveData(OOITask):
    """
    Removes one or multiple data from the given OOI.
    """
    def __init__(self, data):
        """
        :param data: A collection of data to be removed.
        :type data: an object supported by the DataParser class
        """
        self.data_gen = self._parse_data(data)

    def execute(self, OOI):
        """
        Returns the OOI with removed data.

        :param OOI: input OOI
        :type OOI: OOI
        :return: input OOI without the specified data
        :rtype: OOI
        """
        for data_type, data_name in list(self.data_gen(OOI)):
            if data_name is ...:
                OOI.reset_data_type(data_type)
            else:
                del OOI[data_type][data_name]
        return OOI

class RenameData(OOITask):
    """Renames one or multiple data from the given OOI.
    """
    def __init__(self, data):
        """
        :param data: A collection of data to be renamed.
        :type data: an object supported by the DataParser class
        """
        self.data_gen = self._parse_data(data, new_names=True)

    def execute(self, OOI):
        """
        Returns the OOI with renamed data.

        :param OOI: input OOI
        :type OOI: OOI
        :return: input OOI with the renamed data
        :rtype: OOI
        """
        for data_type, data_name, new_data_name in self.data_gen(OOI):
            OOI[data_type][new_data_name] = OOI[data_type][data_name]
            del OOI[data_type][data_name]
        return OOI

class DuplicateData(OOITask):
    """
    Duplicates one or multiple data in an OOI.
    """

    def __init__(self, data, deep_copy=False):
        """
        :param data: A collection of data to be renamed.
        :type data: an object supported by the DataParser class
        :param deep_copy: Make a deep copy of data if set to true, else just assign it.
        :type deep_copy: bool
        """
        self.data_gen = self._parse_data(data, new_names=True)
        self.deep = deep_copy

    def execute(self, OOI):
        """Returns the OOI with copied data.
        :param OOI: Input OOI
        :type OOI: OOI
        :return: Input OOI with the duplicated data.
        :rtype: OOI
        :raises ValueError: Raises an exception when trying to duplicate data 
            with an already existing data name.
        """
        for data_type, data_name, new_data_name in self.data_gen(OOI):
            if new_data_name in OOI[data_type]:
                raise ValueError("A data named '{}' already exists.".\
                                 format(data_type))
            if self.deep:
                OOI[data_type][new_data_name] = copy.deepcopy(OOI[data_type][data_name])
            else:
                OOI[data_type][new_data_name] = OOI[data_type][data_name]
        return OOI

class InitializeData(OOITask):
    """ 
    Initializes the values of a data.
    Example:
        InitializeData((DataType.DATA, 'data1'), 
                       shape=(5, 10, 10, 3), init_value=3)
        # Initialize data of the same shape as (DataType.DATA, 'data1')
        InitializeData((DataType.MASK, 'mask1'), 
                       shape=(DataType.DATA, 'data1'), init_value=1)
    """
    def __init__(self, data, shape, init_value=0, dtype=np.uint8):
        """
        :param data: A collection of data to be renamed.
        :type data: an object supported by the DataParser class
        :param shape: A shape object (t, n, m, d) or a data from which to 
            read the shape.
        :type shape: A tuple or an object supported by the DataParser class
        :param init_value: A value with which to initialize the array of the 
            new data.
        :type init_value: int
        :param dtype: Type of array values.
        :type dtype: NumPy dtype
        :raises ValueError: Raises an exception when passing the wrong shape argument.
        """
        self.features = self._parse_features(features)

        try:
            self.shape_data= next(self._parse_data(shape)())
        except ValueError:
            self.shape_data = None
        if self.shape_data:
            self.shape = None
        elif isinstance(shape, tuple) and len(shape) in (3, 4) and\
            all(isinstance(x, int) for x in shape):
            self.shape = shape
        else:
            raise ValueError("shape argument is not a shape tuple or\
                             a data containing one.")

        self.init_value = init_value
        self.dtype = dtype

    def execute(self, OOI):
        """
        :param OOI: Input OOI.
        :type OOI: OOI
        :return: Input OOI with the initialized additional data.
        :rtype: OOI
        """
        shape = OOI[self.shape_data].shape if self.shape_data else self.shape
        add_data = set(self.data) - set(OOI.get_ooi_items_list())

        for dat in add_data:
            OOI[dat] = np.ones(shape, dtype=self.dtype) * self.init_value
        return OOI

class MoveData(OOITask):
    """ 
    Task to copy/deepcopy fields from one OOI to another.
    """
    def __init__(self, data, deep_copy=False):
        """
        :param data: A collection of data to be renamed.
        :type data: an object supported by the DataParser class
        :param deep_copy: Make a deep copy of data if set to true, else just assign it.
        :type deep_copy: bool
        """
        self.data_gen = self._parse_data(data)
        self.deep = deep_copy

    def execute(self, src_OOI, dst_OOI):
        """
        :param src_OOI: Source OOI from which to take data.
        :type src_OOI: OOI
        :param dst_OOI: Destination OOI to which to move/copy data.
        :type dst_OOI: OOI
        :return: dst_OOI with the additional data from src_OOI.
        :rtype: OOI
        """

        for dat in self.data_gen(src_OOI):
            if self.deep:
                dst_OOI[dat] = copy.deepcopy(src_OOI[dat])
            else:
                dst_OOI[dat] = src_OOI[dat]
        return dst_OOI

class MapDataTask(OOITask):
    """ 
    Applies a function to each dataset in input_data of a patch and stores the 
    results in a set of output_data.
        Example using inheritance:
            class MultiplyData(MapDataTask):
                def map_method(self, f):
                    return f * 2
            multiply = MultiplyData({DataType.DATA: ['f1', 'f2', 'f3']},  # input data
                                        {DataType.MASK: ['m1', 'm2', 'm3']})  # output data
            result = multiply(ooi)
        Example using lambda:
            multiply = MapDataTask({DataType.DATA: ['f1', 'f2', 'f3']},  # input data
                                      {DataType.MASK: ['m1', 'm2', 'm3']},  # output data
                                      lambda f: f*2)   # a function to apply to each data
            result = multiply(ooi)
        Example using a np.max and it's kwargs passed as arguments to the MapDataTask:
            maximum = MapDataTask((DataType.DATA: 'f1'),  # input data
                                     (DataType.MASK, 'm1'),  # output data
                                     np.max, # a function to apply to each data
                                     axis=0) # function's kwargs
            result = maximum(ooi)
    """
    def __init__(self, input_data, output_data, map_function=None, **kwargs):
        """
        :param input_data: A collection of the input data to be mapped.
        :type input_data: an object supported by the DataParser class
        :param output_data: A collection of the output data to which to assign the 
                output data.
        :type output_data: an object supported by the DataParser class
        :param map_function: A function or lambda to be applied to the input data.
        :raises ValueError: Raises an exception when passing data collections with 
            different lengths.
        :param kwargs: kwargs to be passed to the map function.
        """
        self.input_data = list(self._parse_data(input_data))
        self.output_data = list(self._parse_data(output_data))
        self.kwargs = kwargs

        if len(self.input_data) != len(self.output_data):
            raise ValueError('The number of input and output data must match.')

        self.function = map_function if map_function else self.map_method

    def execute(self, OOI):
        """
        :param OOI: Source OOI from which to read the input data.
        :type OOI: OOI
        :return: An OOI with the additional mapped data.
        :rtype: OOI
        """
        for input_data, output_dat in zip(self.input_data , self.output_data):
            OOI[output_dat] = self.function(OOI[input_data], **self.kwargs)
        return OOI

    def map_method(self, data):
        """
        A function that will be applied to the input data.
        """
        raise NotImplementedError('map_method should be overridden.')

class ZipDataTask(OOITask):
    """ 
    Passes a set of input_data to a function, which returns single data as a 
    result and stores it in the given OOI.
        Example using inheritance:
            class CalculateData(ZipDataTask):
                def map_function(self, *f):
                    return f[0] / (f[1] + f[2])
            calc = CalculateData({DataType.DATA: ['f1', 'f2', 'f3']}, # input data
                                     (DataType.MASK, 'm1'))           # output data
            result = calc(ooi)
        Example using lambda:
            calc = ZipDataTask({DataType.DATA: ['f1', 'f2', 'f3']},  # input data
                                  (DataType.MASK, 'm1'),                # output data
                                  lambda f0, f1, f2: f0 / (f1 + f2))     
                                    # a function to apply to each data set
            result = calc(ooi)
        Example using a np.maximum and it's kwargs passed as arguments to the ZipDataTask:
            maximum = ZipDataTask({DataType.DATA: ['f1', 'f2']},  # input data
                                     (DataType.MASK, 'm1'),          # output data
                                     np.maximum,        # a function to apply to each dataset
                                     dtype=np.float64)                  # function's kwargs
            result = maximum(ooi)
    """
    def __init__(self, input_data, output_data, zip_function=None, **kwargs):
        """
        :param input_data: A collection of the input data to be mapped.
        :type input_data: an object supported by the DataParser class
        :param output_data: An output data object to which to assign the the data.
        :type output_data: an object supported by the DataParser class
        :param zip_function: A function or lambda to be applied to the input data.
        :param kwargs: kwargs to be passed to the zip function.
        """
        self.input_data = list(self._parse_data(input_data))
        self.output_data = next(self._parse_data(output_data)())
        self.function = zip_function if zip_function else self.zip_method
        self.kwargs = kwargs

    def execute(self, OOI):
        """
        :param OOI: Source OOI from which to read the data of input data.
        :type OOI: OOI
        :return: An OOI with the additional zipped data.
        :rtype: OOI
        """
        data = [OOI[dat] for dat in self.input_data]
        OOI[self.output_data] = self.function(*data, **self.kwargs)
        return OOI

    def zip_method(self, *f):
        """ 
        A function that will be applied to the input data if overridden.
        """
        raise NotImplementedError('zip_method should be overridden.')

class MergeDataTask(ZipDataTask):
    """ 
    Merges multiple data together by concatenating data along the last axis.
    """
    def zip_method(self, *f):
        """
        Concatenates the data of features along the last axis.
        """
        return np.concatenate(f, axis=-1)

class ExtractBandsTask(MergeDataTask):
    """ 
    Moves a subset of bands from one set of data to a new one.
    """
    def __init__(self, input_data, output_data, bands):
        """
        :param input_data: A source data from which to take the subset of bands.
        :type input_data: an object supported by the DataParser class
        :param output_data: Output data to which to write the bands.
        :type output_data: an object supported by the DataParser class
        :param bands: A list of bands to be moved.
        :type bands: list
        """
        super().__init__(input_data, output_data)
        self.bands = bands

    def map_method(self, data):
        if not all(band < data.shape[-1] for band in self.bands):
            raise ValueError("Band index out of data dimensions.")
        return data[..., self.bands]

class CreateOOITask(OOITask):
    """
    Creates an OOI
    """
    def execute(self, **kwargs):
        """
        Returns a newly created OOI with the given kwargs.
        :param kwargs: Any valid kwargs accepted by OOI class
        :return: A new OOI.
        :rtype: OOI
        """
        return OOI(**kwargs)

class MergeOOIsTask(OOITask):
    """ 
    Merge content from multiple OOIs into a single OOI
    """
    def __init__(self, **merge_kwargs):
        """
        :param merge_kwargs: A keyword arguments defined for OOI.merge method
        """
        self.merge_kwargs = merge_kwargs

    def execute(self, *OOIs):
        """
        :param OOIes: OOIes to be merged
        :type OOIes: OOI
        :return: A new OOI with merged content
        :rtype: OOI
        """
        if not OOIs:
            raise ValueError('At least one OOI should be given')
        return OOIs[0].merge(*OOIs[1:], **self.merge_kwargs)

