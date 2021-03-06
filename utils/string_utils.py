# -*- coding: utf-8 -*-
"""
Created on Fri May 14 13:52:51 2021

@author: freeridingeo
"""

import re

def string_to_variable(string, extension=None):
    """
    :param string: string to be used as python variable name
    :type string: str
    :param extension: string to be appended to string
    :type extension: str
    :return: valid python variable name
    :rtype: str
    """
    string = re.sub('[^0-9a-zA-Z_]', '', string)
    string = re.sub('^[^a-zA-Z_]+', '', string)
    if extension:
        string += extension
    return string

def parse_literal(x):
    """
    return the smallest possible data type for a string or list of strings

    x: str or list
        a string to be parsed
    Returns int, float or str
        the parsing result
    
    Examples
    --------
    >>> isinstance(parse_literal('1.5'), float)
    >>> isinstance(parse_literal('1'), int)
    >>> isinstance(parse_literal('foobar'), str)
    """
    if isinstance(x, list):
        return [parse_literal(y) for y in x]
    elif isinstance(x, (bytes, str)):
        try:
            return int(x)
        except ValueError:
            try:
                return float(x)
            except ValueError:
                return x
    else:
        raise TypeError('input must be a string or a list of strings')


