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
from scipy.stats import linregress, spearmanr

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
from matplotlib.ticker import MultipleLocator
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
site_name = 'Ca-Obs'


MOD13_file = f'/uufs/chpc.utah.edu/common/home/lin-group23/ltk/MODIS/NDVI_EVI_MOD13A2/from_GEE/{site_name}/modis_ndvi_timeseries_{site_name}.csv'
MCD43_file = f'/uufs/chpc.utah.edu/common/home/lin-group23/ltk/MODIS/Reflectance_MCD43A4/from_GEE/{site_name}/mcd43a4_reflectance_{site_name}.csv'

MOD13_outdir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/data/QC_PhotoSpec_sites_backup/MOD13'
MCD43_outdir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/data/QC_PhotoSpec_sites_backup/MCD43'

os.makedirs(MOD13_outdir, exist_ok=True)
os.makedirs(MCD43_outdir, exist_ok=True)

MOD13_outfile = os.path.join(MOD13_outdir, f'MOD13_pt_{site_name}.nc')
MCD43_outfile = os.path.join(MCD43_outdir, f'MCD43_pt_{site_name}.nc')

#########################################
# Define Global functions
#########################################


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
    },
    'US-Ne3': {
        'coords': (-96.4397, 41.1797),
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


MOD13_df = pd.read_csv(MOD13_file)
print("MOD13_df columns:", MOD13_df.columns.tolist())
MOD13_times = pd.to_datetime(MOD13_df['date'], format='%Y-%m-%d')
# Filter MOD13_df for composite_years and NDVI >= 0.5
MOD13_df = MOD13_df[MOD13_times.dt.year.isin(composite_years)].reset_index(drop=True)
MOD13_df = MOD13_df[MOD13_df['ndvi'] >= 0.5].reset_index(drop=True)
MOD13_times = pd.to_datetime(MOD13_df['date'], format='%Y-%m-%d')


MCD43_df = pd.read_csv(MCD43_file)
print("MCD43_df columns:", MCD43_df.columns.tolist())
MCD43_times = pd.to_datetime(MCD43_df['date'], format='%Y-%m-%d')

# Resample MCD43_df to 8-day resolution
MCD43_df['date'] = MCD43_times
MCD43_df_8day = MCD43_df.set_index('date').resample('8D').mean(numeric_only=True).reset_index()
MCD43_times_8day = pd.to_datetime(MCD43_df_8day['date'])

MOD13_pt = xr.Dataset(
    {
        'NDVI': (['time'], MOD13_df['ndvi'].values),
        'view_zenith': (['time'], MOD13_df['view_zenith'].values),
        'solar_zenith': (['time'], MOD13_df['solar_zenith'].values),
    },
    coords={
        'time': MOD13_times.values,
    }
)

MCD43_pt = xr.Dataset(
    {
        'NDVI': (['time'], MCD43_df_8day['ndvi'].values),
        'band1_red': (['time'], MCD43_df_8day['band1_red'].values),
        'band2_nir': (['time'], MCD43_df_8day['band2_nir'].values),
    },
    coords={
        'time': MCD43_times_8day.values,
    }
)

# MOD13_preexisting = xr.open_dataset(MOD13_outfile)
# MCD43_preexisting = xr.open_dataset(MCD43_outfile)


MOD13_pt.to_netcdf(MOD13_outfile)
print(f"Saved MOD13 data to {MOD13_outfile}")

MCD43_pt.to_netcdf(MCD43_outfile)
print(f"Saved MCD43 data to {MCD43_outfile}")

# %%
