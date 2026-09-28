#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Lewis Kunik - University of Utah
# Contact: lewis.kunik@utah.edu
#
# Created on Tue Oct 24 2023
#
#

#########################################
# Import packages
#########################################
#%%
# File system packages
import os  # operating system library
import sys
import glob
from pathlib import Path
import copy

# runtime packages
import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
# warnings.simplefilter(action='ignore', category=DeprecationWarning)
import gc

# numerical/data management packages
import math
import numpy as np
import xarray as xr  # for multi dimensional data
# import rioxarray as rxr
import pandas as pd
import xesmf as xe
import struct
from rasterio.enums import Resampling
import geopandas as gpd

# Stats packages
from scipy.stats import linregress

# shapefile/geospatial packages

# time and date packages
import time
from datetime import datetime as dt  # date time library
from datetime import timedelta

# plotting packages
from matplotlib import pyplot as plt  # primary plotting module
from shapely.geometry import mapping
import cftime
import random
from matplotlib.patches import Patch
from matplotlib.dates import MonthLocator, DateFormatter
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter as FFmt  # need this for formatting plot ticks
import cartopy.crs as ccrs  #
from cartopy.io import img_tiles  # cartopy's implementation of webtiles
import cartopy.io.shapereader as shpreader
import cartopy.feature as cfeature


#########################################
# Define Global Filepaths
#########################################


#%%
site_name = 'US-UMB'


FloX_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/data/FloX/US-Ne3/'
FloX_2018_file = os.path.join(FloX_dir, f'DFlox_SIF_VIs_2018.csv')
FloX_2019_file = os.path.join(FloX_dir, f'DFlox_SIF_VIs_2019.csv')




#########################################
# Define Global functions
#########################################

def remove_running_window_outliers(df, window='14D', value_columns=('CCI', 'NDVI')):
    """Remove rows containing values outside running 1st/99th percentiles."""
    df = df.sort_values('DateTime').copy()
    indexed = df.set_index('DateTime')
    keep = pd.Series(True, index=indexed.index)

    for column in value_columns:
        if column not in indexed:
            continue
        values = indexed[column]
        lower = values.rolling(window, center=True, min_periods=2).quantile(0.01)
        upper = values.rolling(window, center=True, min_periods=2).quantile(0.99)
        keep &= values.isna() | ((values >= lower) & (values <= upper))

    return indexed.loc[keep.to_numpy()].reset_index()


####################################
### Set up map parameters
####################################


#########################################
# Define Global Variables and constants
#########################################

# composite_years = np.arange(2017, 2019)
# site_lon, site_lat = [-105.5464, 	40.0329]


# Site coordinates, date ranges, and save paths
site_info = {
    'US-NR1': {
        'coords': (-105.5464, 40.0329),
        'start': '2017-01-01',
        'end': '2018-12-31'
    },
    'OSBS': {
        'coords': (-81.993431, 29.689282),
        'start': '2021-01-01',
        'end': '2022-12-31'
    },
    'Ca-Obs': {
        'coords': (-105.1178, 53.9872),
        'start': '2018-01-01',
        'end': '2022-12-31'
    },
    'DEJU': {
        'coords': (-145.75136, 63.88112),
        'start': '2019-01-01',
        'end': '2021-12-31'
    },
    'US-UMB': {
        'coords': (-84.7138, 45.5598),
        'start': '2018-01-01',
        'end': '2019-12-31'
    }
}

site_lon = site_info[site_name]['coords'][0]
site_lat = site_info[site_name]['coords'][1]
composite_years = np.arange(int(site_info[site_name]['start'][:4]), int(site_info[site_name]['end'][:4]) + 1)

#########################################
# Begin main
#########################################
#%%
# def main():

# mark start time to keep track of elapsed
start_total = time.time()



#%%

# # Read the 2018 FloX workbook into a pandas DataFrame.
# FloX_df_2018 = pd.ExcelFile(FloX_2018_file)
# dfs = {sheet_name: FloX_df_2018.parse(sheet_name) 
#           for sheet_name in FloX_df_2018.sheet_names}

# FloX_df_2018 = pd.read_excel(FloX_2018_file)
FloX_df_2018 = pd.read_csv(FloX_2018_file)


print("FloX_df_2018 columns:", FloX_df_2018.columns.tolist())
print(f'FloX_df_2018 rows (before filtering for valid datetimes): {len(FloX_df_2018)}')
FloX_df_2018 = FloX_df_2018[FloX_df_2018['DoY'].between(1, 366)]
FloX_df_2018['DateTime'] = pd.to_datetime(FloX_df_2018['DateTime'])
FloX_df_2018 = FloX_df_2018[FloX_df_2018['DateTime'].between('2018-01-01', '2018-12-31')]
FloX_df_2018 = FloX_df_2018[
    FloX_df_2018['DateTime'].dt.time.between(pd.Timestamp('13:00').time(), pd.Timestamp('14:00').time())
]

FloX_df_2018 = FloX_df_2018[FloX_df_2018['NDVI'].between(0, 1)]
FloX_df_2018 = remove_running_window_outliers(FloX_df_2018)




print(f'FloX_df_2018 rows (after filtering for valid datetimes): {len(FloX_df_2018)}\n\n\n')

FloX_df_2019 = pd.read_csv(FloX_2019_file)
print(f'FloX_df_2019 rows (before filtering for valid datetimes): {len(FloX_df_2019)}')
FloX_df_2019['DoY'] = FloX_df_2019['DoY'] - 365
FloX_df_2019 = FloX_df_2019[FloX_df_2019['DoY'].between(1, 366)]
FloX_df_2019['DateTime'] = pd.to_datetime(FloX_df_2019['DateTime'])
FloX_df_2019 = FloX_df_2019[FloX_df_2019['DateTime'].between('2019-01-01', '2019-12-31')]
FloX_df_2019 = FloX_df_2019[
    FloX_df_2019['DateTime'].dt.time.between(pd.Timestamp('13:00').time(), pd.Timestamp('14:00').time())
]
FloX_df_2019 = FloX_df_2019[FloX_df_2019['NDVI'].between(0, 1)]
FloX_df_2019 = remove_running_window_outliers(FloX_df_2019)
print(f'FloX_df_2019 rows (after filtering for valid datetimes): {len(FloX_df_2019)}')


FloX_df = pd.concat([FloX_df_2018, FloX_df_2019], ignore_index=True)



### PLOT 2018 Data
fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(FloX_df_2018['DateTime'].values, FloX_df_2018['CCI'].values, s=10, color='black', alpha=0.7)
# ax.set_xlabel('DOY')
ax.set_ylabel('CCI')
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(FloX_df_2018['DateTime'].values, FloX_df_2018['NDVI'].values, s=10, color='black', alpha=0.7)
# ax.set_xlabel('DOY')
ax.set_ylabel('NDVI')
plt.tight_layout()
plt.show()





### PLOT 2019 Data
fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(FloX_df_2019['DateTime'].values, FloX_df_2019['CCI'].values, s=10, color='black', alpha=0.7)
# ax.set_xlabel('DOY')
ax.set_ylabel('CCI')
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(FloX_df_2019['DateTime'].values, FloX_df_2019['NDVI'].values, s=10, color='black', alpha=0.7)
# ax.set_xlabel('DOY')
ax.set_ylabel('NDVI')
plt.tight_layout()
plt.show()

