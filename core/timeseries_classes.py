# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 13:16:21 2021

@author: freeridingeo
"""

from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime as dt, timedelta

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

    def prepare_timeseries_yr_m_day_doyr(self):
        
        self.data_df = self.data_df.assign(
            date = lambda x: pd.to_datetime(x['date']), 
            year = lambda x: x['date'].dt.year,
            month = lambda x: x['date'].dt.month,
            day = lambda x: x['date'].dt.day,
            dayofyear = lambda x: x['date'].dt.dayofyear
            )
        return self.data_df
        
    def extract_monthofyear(self, columname):
        self.data_df['Month_Year'] =\
            self.data_df.index.map(lambda d: d.strftime('%m-%Y'))
        monthly_stats = self.data_df.groupby(by='Month_Year')[columname].\
                    aggregate([np.mean, np.median, np.std])
        monthly_stats.reset_index(inplace=True)
        monthly_stats['Year'] = monthly_stats['Month_Year']\
                        .map(lambda m: pd.to_datetime(m, format='%m-%Y').strftime('%Y'))
        monthly_stats['Month'] = monthly_stats['Month_Year']\
                        .map(lambda m: pd.to_datetime(m, format='%m-%Y').strftime('%m'))
        monthly_stats.sort_values(by=['Year', 'Month'], inplace=True)
        monthly_stats.index = monthly_stats['Month_Year']
    
        fig = plt.figure(figsize=(15, 9))
        ax = fig.add_subplot(1,1,1)
        monthly_stats['mean'].plot(ax=ax, color='b', label="mean")
        monthly_stats['std'].plot(ax=ax, color='r', label="std")
        ax.set_title('Monthly statistics: Mean (blue) & Std. Dev. (red)')
        plt.legend(loc="best")
        plt.xticks(rotation=45)
        
        return self.data_df, monthly_stats

    def moving_average(self, columname, stridelength=7):
        moving_average = self.data_df[columname].rolling(stridelength).mean()
        
        fig = plt.figure(figsize=(5.5, 5.5))
        ax = fig.add_subplot(2,1,1)
        self.data_df[columname].plot(ax=ax, color='b')
        ax.set_title('Unsmoothed')
        ax = fig.add_subplot(2,1,2)
        moving_average.plot(ax=ax, color='r')
        ax.set_title(stridelength, ' Moving Average')
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=2.0)
        return moving_average

    def resample_column(self, columname, freq='MS'):
        y = self.data_df.resample(freq).mean()

        plt.figure(figsize=(20, 8))
        plt.plot(y.index, y, 'b-', label = columname)
        plt.xlabel('Date'); plt.title('Monthly Resampled Time Series')
        plt.legend()

    def subsample_column(self, start_date, end_date):
        time = pd.date_range(start_date, end_date)

        
    