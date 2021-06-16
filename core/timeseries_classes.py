# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 13:16:21 2021

@author: freeridingeo
"""

from pathlib import Path
import pandas as pd
import numpy as np


import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('bmh')


class TimeSeries(object):

    def __init__(self, path):
        if isinstance(path, str):
            self.path = Path(path)
        elif isinstance(path, Path):
            self.path = path
    
        self.ext = self.path.suffix
        
    def read_file(self, separation = "\t"):
        if self.ext in [".csv", ".txt"]:
            self.data_df = pd.read_csv(self.path, sep = separation)
    
        return self.data_df

    def get_column_names(self):
        columnames = self.data_df.columns
        if not columnames:
            len_cols = len(self.data_df)
            columnames = [str(col) for col in np.arange(len_cols)]
        return columnames

    def plot_specific_column(self, columname, timeunit = "month"):
        plt.figure(figsize=(16,5))
        plt.plot(self.data_df.index, self.data_df, color='tab:blue')
        plt.gca().set(xlabel="Time", ylabel=columname)
        plt.show()

        fig, axes = plt.subplots(1, 2, figsize=(20,7), dpi= 80)
        sns.boxplot(x='year', y=columname, data=self.data_df, ax=axes[0])
        axes[0].set_title('Year-wise Box Plot\n(The Trend)', fontsize=18)
        axes[1].set_title('Month-wise Box Plot\n(The Seasonality)', fontsize=18)
        plt.show()

    def resample_column(self, columname, freq='MS'):
        y = self.data_df.resample(freq).mean()

        plt.figure(figsize=(20, 8))
        plt.plot(y.index, y, 'b-', label = columname)
        plt.xlabel('Date'); plt.title('Monthly Resampled Time Series')
        plt.legend()


        
    