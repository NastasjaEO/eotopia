# -*- coding: utf-8 -*-
"""
Created on Sat May 22 11:14:07 2021

@author: freeridingeo
"""

import os
from pathlib import Path


def configure_basic_workspace(path, **kwargs):
    
    folder_list = []

    MAIN_FOLDER = Path(path)
    INPUT_DATA_FOLDER = MAIN_FOLDER / 'InputData'
    RASTER_DATA = INPUT_DATA_FOLDER / 'RasterData'
    VECTOR_DATA = INPUT_DATA_FOLDER / 'VectorData'
    OUTPUT_DATA_FOLDER = MAIN_FOLDER / 'Output'

    folder_list.append((MAIN_FOLDER, INPUT_DATA_FOLDER, RASTER_DATA, VECTOR_DATA,
                        OUTPUT_DATA_FOLDER))

    if not os.path.exists(OUTPUT_DATA_FOLDER):
        OUTPUT_DATA_FOLDER.mkdir()
        
    if "classification" in kwargs.values():
        RULES_PATH = OUTPUT_DATA_FOLDER / 'Rules'
        if not os.path.exists(RULES_PATH):
            RULES_PATH.mkdir()
        SAMPLES_PATH = OUTPUT_DATA_FOLDER  / 'Samples'
        if not os.path.exists(SAMPLES_PATH):
            SAMPLES_PATH.mkdir()
        MODELS_PATH = OUTPUT_DATA_FOLDER  / 'Models'
        if not os.path.exists(MODELS_PATH):
            MODELS_PATH.mkdir()
        PREDICTIONS_PATH = OUTPUT_DATA_FOLDER  / 'Predictions'
        if not os.path.exists(PREDICTIONS_PATH):
            PREDICTIONS_PATH.mkdir()

    if "eopatch" in kwargs.values():
        EOPATCH_PATH = INPUT_DATA_FOLDER / 'EOPatches'
        if not os.path.exists(EOPATCH_PATH):
            EOPATCH_PATH.mkdir()
        folder_list.append(EOPATCH_PATH)


    if "eopatch" and "classification" in kwargs.values():
        EOPATCH_PATH_SAMPLES = EOPATCH_PATH / 'EOPatche_Samples'
        if not os.path.exists(EOPATCH_PATH_SAMPLES):
            EOPATCH_PATH_SAMPLES.mkdir()
    
    return folder_list