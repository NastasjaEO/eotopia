# -*- coding: utf-8 -*-
"""
Created on Wed May 12 14:07:28 2021

@author: freeridingeo
"""

def add_metadata_to_eopatch(eopatch, metainfo):
    eopatch.meta_info[str(metainfo)] = metainfo
    return eopatch

def get_eopatch_size(eopatch):
    return eopatch.meta_info.get('size_x'), eopatch.meta_info.get('size_y')

def get_eopatch_bbox(eopatch):
    return eopatch.bbox


