# -*- coding: utf-8 -*-
"""
Created on Mon May 24 11:03:32 2021

@author: freeridingeo
"""

def dissolve(inlist):
    """
    list and tuple flattening
    
    Parameters
    ----------
    inlist: list
        the list with sub-lists or tuples to be flattened
    
    Returns
    -------
    list
        the flattened result
    
    Examples
    --------
    >>> dissolve([[1, 2], [3, 4]])
    [1, 2, 3, 4]
    
    >>> dissolve([(1, 2, (3, 4)), [5, (6, 7)]])
    [1, 2, 3, 4, 5, 6, 7]
    """
    out = []
    for i in inlist:
        i = list(i) if isinstance(i, tuple) else i
        out.extend(dissolve(i)) if isinstance(i, list) else out.append(i)
    return out

