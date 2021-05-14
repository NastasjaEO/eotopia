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
