# -*- coding: utf-8 -*-
"""
Created on Fri May 21 12:09:21 2021

@author: freeridingeo
"""

import os
import numpy as np

from PIL import Image

import sys
sys.path.append("D:/Code/eotopia/repo_core")
from eodata import EOPatch


def rgb_color_scale(arr):
    """correct the RGB bands to be a composed bands of value between 0 255 
    for visualization purpose

    Args:
        arr: RGB bands in numpy array 
    Return:
        arr: numpy array that values range from 0 to 255 for visualization
    """
    str_arr = (arr + 1) * 127.5
    return str_arr

def save_rgb_2_png(patch, feature_name, inference=False):
    """Save RGB, spectral info and labeled images for the coming deep learning
    
    Args:
        patch: Saved eopatches for deep learning LULC training and prediction 
    Return:
        None: images in the PNG.
    """
    
    patch_data = EOPatch.load(patch)
    bands = patch_data.data['FEATURES']
    patch_dir, patch_id = patch.split("/")
    if inference == True:
        inds_path = "inds_inference"

        if not os.path.isdir(inds_path):
            os.makedirs(inds_path)
        for i in range(len(bands)):
            #get NDVI, NDWI and NDBI from the bands
            inds = rgb_color_scale(bands[i][..., [-3, -2, -1]]).astype("uint8")
            Image.fromarray(inds)\
                .save(os.path.join(inds_path, patch+'_'+str(i)+'.png'))

    else:
        feature = patch_data.mask_timeless[feature_name]
        
        feature_path = "feature_all"
        rgb_path = "rgb_all"
        inds_path = "inds_all"
        if not os.path.isdir(feature_path):
            os. makedirs(feature_path)
        if not os.path.isdir(rgb_path):
            os.makedirs(rgb_path)
        if not os.path.isdir(inds_path):
            os.makedirs(inds_path)
        for i in range(len(bands)):
            # Switch R, G, B band index from bands
            rgb = rgb_color_scale(bands[i][..., [2, 1, 0]]).astype("uint8")
            inds = np.clip(rgb_color_scale(bands[i][..., [-3, -2, -1]])\
                           .astype("uint8"), 0, 255)
            Image.fromarray(rgb).save(os.path.join(rgb_path, "{}_{}.png"\
                                                   .format(patch_id, str(i))))
            Image.fromarray(inds)\
                .save(os.path.join(inds_path, "{}_{}.png"\
                                   .format(patch_id, str(i))))
            Image.fromarray(feature.astype("uint8").squeeze(-1), 'L')\
                .save(os.path.join(feature_path, "{}_{}.png"\
                                   .format(patch_id, str(i))))

