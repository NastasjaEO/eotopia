# -*- coding: utf-8 -*-
"""
Created on Fri May 21 10:49:51 2021

@author: freeridingeo
"""

from sentinelhub import DataCollection


def list_sentinelhub_data():
    for collection in DataCollection.get_available_collections():
        print(collection)

    