# -*- coding: utf-8 -*-
"""
Created on Fri May  7 22:23:59 2021

@author: nasta
"""


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