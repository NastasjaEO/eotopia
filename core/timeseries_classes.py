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

def cum_mean(arr):
    cum_sum = np.cumsum(arr, axis=0)    
    for i in range(cum_sum.shape[0]):       
        if i == 0:
            continue        
        cum_sum[i] =  cum_sum[i] / (i + 1)
    return cum_sum

def cum_sum(arr):
    cum_sum = np.cumsum(arr, axis=0)    
    for i in range(cum_sum.shape[0]):       
        if i == 0:
            continue        
    return cum_sum



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

        fig = plt.figure(figsize=(20,8))
        self.data_df.groupby(['Date'])[columname].sum().plot(figsize=(10,6), style='o')
        plt.xlabel("Time")
        plt.ylabel(columname)
        plt.title("Scattered values of "+str(columname))
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

    def timeseries_statistics(self, columname,ndv=0): 
        tsmetrics={}
        rperc = np.nanpercentile(self.data_df[columname], [5,50,95])
        tsmetrics['mean']=np.nanmean(self.data_df[columname])
        tsmetrics['max']=np.nanmax(self.data_df[columname])
        tsmetrics['min']=np.nanmin(self.data_df[columname])
        tsmetrics['range']=tsmetrics['max']-tsmetrics['min']
        tsmetrics['median']=rperc[1]
        tsmetrics['p5']=rperc[0]
        tsmetrics['p95']=rperc[2]
        tsmetrics['prange']=rperc[2]-rperc[0]
        tsmetrics['var']=np.nanvar(self.data_df[columname])
        tsmetrics['cov']=tsmetrics['var']/tsmetrics['mean']

        fig, ax= plt.subplots(1,2,figsize=(16,4))
        ax[0].hist(tsmetrics['var'].flatten(),bins=100)
        ax[1].hist(tsmetrics['cov'].flatten(),bins=100)
        _=ax[0].set_title('Variance')
        _=ax[1].set_title('Coefficient of Variation')
        
        metric_keys=['mean', 'median', 'max', 'min', 
             'p95', 'p5','range', 'prange','var','cov']
        fig= plt.figure(figsize=(16,40))
        idx=1
        for i in tsmetrics.keys:
            ax = fig.add_subplot(5,2,idx)
            if i=='var': vmin,vmax=(0.0,0.005)
            elif i == 'cov': vmin,vmax=(0.,0.04)
            else:
                vmin,vmax=(0.0001,0.3)
                ax.imshow(tsmetrics[i],vmin=vmin,vmax=vmax,cmap='gray')
                ax.set_title(i.upper())
                ax.axis('off')
                idx+=1

        return tsmetrics

    def timeseries_mean_cumsum(self, columname):
        """
        Calculates the cummulative sum of the mean of a timeseries
        Input:
        """
        timeseries_mean_cumsum = cum_sum(self.data_df[columname])
        fig = plt.figure(figsize=(20,15))
        plt.plot(self.data_df.index, timeseries_mean_cumsum, linewidth=5)
        plt.title(columname, fontsize=23)
        plt.show()
        return timeseries_mean_cumsum

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
        y = self.data_df[columname].resample(freq).mean()

        plt.figure(figsize=(20, 8))
        plt.plot(y.index, y, 'b-', label = columname)
        plt.xlabel('Date'); plt.title('Monthly Resampled Time Series')
        plt.legend()

    def subsample_column(self, columname, start_date, end_date, 
                         temp_scale="month"):
        t1 = start_date
        t2 = end_date

        if temp_scale == "month":
            df_sub = self.data_df[columname][np.logical_and(self.data_df.index.month>=t1, 
                                                          self.data_df.index.month<=t2)]

        elif temp_scale == "year":
            df_sub = self.data_df[columname][np.logical_and(self.data_df.index.year>=t1, 
                                                            self.data_df.index.year<=t2)]

        elif temp_scale == None:
            df_sub = self.data_df[columname][np.logical_and(self.data_df.index>=t1, 
                                                            self.data_df.index<=t2)]

        fig, ax = plt.subplots(figsize=(16,4))
        df_sub.plot(ax=ax)
        plt.ylabel(columname)
        _=plt.legend([str(t1) + " to " + str(t2)])
        
        
    