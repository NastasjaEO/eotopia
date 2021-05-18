# -*- coding: utf-8 -*-
"""
Created on Tue May 18 09:24:27 2021

@author: freeridingeo
"""

import enum

class CROPTYPECLASS(enum.Enum):
    NO_DATA = (0, 'No Data', 'white')
    Beets = (1, 'Beets', 'orange')
    Meadows = (2, 'Meadows', 'black')
    Fallow_land = (3, 'Fallow land', 'xkcd:azure')
    Peas = (4, 'Peas', 'xkcd:salmon')
    Pasture = (5, 'Pasture', 'xkcd:navy')
    Hop = (6, 'Hop', 'xkcd:lavender')
    Grass = (7, 'Grass', 'xkcd:lightblue')
    Poppy = (8, 'Poppy', 'xkcd:brown')
    Winter_rape = (9, 'Winter rape', 'xkcd:shit')
    Maize = (10, 'Maize', 'xkcd:beige')
    Winter_cereals = (11, 'Winter cereals', 'xkcd:apricot')
    LL_ao_GM = (12, 'LL and/or GM', 'crimson')
    Pumpkins = (13, 'Pumpkins', 'lightgrey')
    Soft_fruit = (14, 'Soft fruit', 'firebrick')
    Summer_cereals = (15, 'Summer cereals', 'xkcd:grey')
    Sun_flower = (16, 'Sun flower', 'xkcd:jade')
    Vegetables = (17, 'Vegetables', 'xkcd:ultramarine')
    Buckwheat = (18, 'Buckwheat', 'xkcd:tan')
    Alpine_Meadows = (19, 'Alpine meadows', 'xkcd:lime')
    Potatoes = (20, 'Potatoes', 'pink')
    Beans = (21, 'Beans', 'xkcd:darkgreen')
    Vineyards = (22, 'Vineyards', 'magenta')
    Other = (23, 'Other', 'xkcd:gold')
    Soybean = (24, 'Soybean', 'xkcd:clay')
    Orchards = (25, 'Orchards', 'olivedrab')
    Multi_use = (26, 'Multi use', 'orangered')

    def __init__(self, val1, val2, val3):
        self.id = val1
        self.class_name = val2
        self.color = val3

