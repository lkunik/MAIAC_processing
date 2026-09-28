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
site_name = 'US-Ne3'

MOD09_basedir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/MODIS/Reflectance_MOD09/CV-MVC/QCfilt_CloudFree_LowAOD_ClearAdj'

out_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/data/QC_PhotoSpec_sites_backup/MOD09/'
os.makedirs(out_dir, exist_ok=True)

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


### Load MOD09 data for the same site and time period

MOD09_dir = f'{MOD09_basedir}/{site_name}/'
MOD09_files = sorted(glob.glob(os.path.join(MOD09_dir, '*.nc')))

MOD09_arr = []
for MOD09_file in MOD09_files:

    print(f"Loading MOD09 file: {MOD09_file}")
    MOD09_xr = xr.open_dataset(MOD09_file)
    if 'x' in MOD09_xr.dims:
        MOD09_xr = MOD09_xr.squeeze(['x'], drop = True)
    if 'y' in MOD09_xr.dims:
        MOD09_xr = MOD09_xr.squeeze(['y'], drop = True)
    MOD09_arr.append(MOD09_xr)

if len(MOD09_arr) > 0:
    MOD09_pt = xr.concat(MOD09_arr, dim='time', coords='minimal', compat='override').sortby('time')
    MOD09_out_file = os.path.join(out_dir, f'MOD09_pt_{site_name}.nc')
    MOD09_pt.to_netcdf(MOD09_out_file)
    print(f"Saved MOD09 point data to {MOD09_out_file}")
else:
    print("No MOD09 data found for the specified years and site.")

# %%
