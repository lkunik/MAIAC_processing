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
site_name = 'US-UMB'
MODIS_tile = 'h12v04'

GCOM_dir_QCbasic = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/GCOM-C/RSRF/CV-MVC/pre-fill/QCfilt_basic/{MODIS_tile}/'

out_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/data/QC_PhotoSpec_sites_backup/GCOM/'
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

def clip_2dcurv_ds_250m_to_point(ds, lon_pt, lat_pt):
    # Find the x, y indices where lon_pt and lat_pt are found in ds
    lon_2d = ds.lon.values
    lat_2d = ds.lat.values

    # Compute the absolute difference and find the indices of the minimum
    dist = np.abs(lon_2d - lon_pt) + np.abs(lat_2d - lat_pt)
    y_idx, x_idx = np.unravel_index(np.argmin(dist), lon_2d.shape)
    lon_selected = ds.lon.isel(x=x_idx, y=y_idx).values
    lat_selected = ds.lat.isel(x=x_idx, y=y_idx).values
    print(f"Nearest point in dataset to ({lon_pt}, {lat_pt}) is at ({lon_selected}, {lat_selected}) with distance {dist[y_idx, x_idx]:.6f}")

    ds_point_2d = ds.isel(
        x=slice(max(0, x_idx - 1), min(x_idx + 2, lon_2d.shape[0])),
        y=slice(max(0, y_idx - 1), min(y_idx + 2, lat_2d.shape[1]))
    )
    ds_point = ds_point_2d.mean(dim=['x', 'y'], skipna=True)
    return ds_point

### Load all the MCD19 data


GCOM_arr = []
GCOM_files = sorted(glob.glob(os.path.join(GCOM_dir_QCbasic, '*.nc')))

for composite_year in composite_years:
    print(f"Loading GCOM SGLI RSRF year {composite_year}...")

    
    for composite_nn in range(0, 23):
        print(f"  composite {composite_nn:02d}...")
        # Sample file: GC1SG1_RSRF_h10v04_250m_16day_CV-MVC_QCfilt_basic_2019-20.nc
        GCOM_file = os.path.join(GCOM_dir_QCbasic, f'GC1SG1_RSRF_{MODIS_tile}_250m_16day_CV-MVC_QCfilt_basic_{composite_year}-{composite_nn:02d}.nc')

        if not os.path.exists(GCOM_file):
            print(f"File does not exist: {GCOM_file}")
            continue
        GCOM_ds_curv = xr.open_dataset(GCOM_file)
        GCOM_ds_pt = clip_2dcurv_ds_250m_to_point(GCOM_ds_curv, site_lon, site_lat)

        # Find the x, y indices where lon_pt and lat_pt are found in ds
        ds = GCOM_ds_curv
        lon_2d = ds.lon.values
        lat_2d = ds.lat.values

        # Compute the absolute difference and find the indices of the minimum
        dist = np.abs(lon_2d - site_lon) + np.abs(lat_2d - site_lat)
        x_idx, y_idx = np.unravel_index(np.argmin(dist), lon_2d.shape)
        ds_point_2d = ds.isel(x=slice(x_idx-1, x_idx+2), y=slice(y_idx-1, y_idx+2))
        ds_point = ds_point_2d.mean(dim=['x', 'y'], skipna=True)
        GCOM_arr.append(GCOM_ds_pt)

# Compile arrays and compute VIs
if len(GCOM_arr) > 0:
    GCOM_pt = xr.concat(GCOM_arr, dim='time')
    GCOM_pt['time'] = pd.to_datetime(GCOM_pt.time.values)
    GCOM_pt = GCOM_pt.assign(NDVI = (GCOM_pt.VN11 - GCOM_pt.VN08) / (GCOM_pt.VN11 + GCOM_pt.VN08))
    GCOM_pt = GCOM_pt.assign(CCI = (GCOM_pt.VN05 - GCOM_pt.VN08) / (GCOM_pt.VN05 + GCOM_pt.VN08))

    GCOM_out_file = os.path.join(out_dir, f'GCOM_pt_{site_name}.nc')
    GCOM_pt.to_netcdf(GCOM_out_file)
    print(f"Saved GCOM point data to {GCOM_out_file}")
else:
    print("No GCOM data found for the specified years and site.")
