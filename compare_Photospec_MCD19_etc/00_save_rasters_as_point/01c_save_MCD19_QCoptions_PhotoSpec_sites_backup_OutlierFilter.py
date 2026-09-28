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

#########################################
# Define Global functions
#########################################

def floor(value, decimal_digits):
  scale = (10**decimal_digits)
  return  np.floor(value * scale)/scale

def ceil(value, decimal_digits):
  scale = (10**decimal_digits)
  return  np.ceil(value * scale)/scale


def binary(num):
    return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', num))

binary_v = np.vectorize(binary)

def get_bits(str, start_bit, end_bit):
   str_split = [*str][start_bit:end_bit+1]
   return ''.join(str_split)

get_bits_v = np.vectorize(get_bits)

tanh_v = np.vectorize(np.tanh)

arccos_v = np.vectorize(np.arccos)

def rad_to_deg(rad):
   return rad*180/math.pi

def perdelta(start, end, delta):
    curr = start
    while curr < end:
        yield curr
        curr += delta

def td_min(td):
    return td.seconds//60

def td_sec(td):
    return td.seconds%60


def parse_timestamps_to_datetime_from_global_attrs(orbit_time_stamp):
    timestamps_raw = orbit_time_stamp.strip().split()
    orbit_times = []
    for timestamp in timestamps_raw:
        year = int(timestamp[0:4])
        julian_day = int(timestamp[4:7])
        hour = int(timestamp[7:9])
        minute = int(timestamp[9:11])
        dt_obj = dt(year, 1, 1) + timedelta(days=julian_day - 1, hours=hour, minutes=minute)
        orbit_times.append(dt_obj)
    # Convert to numpy.datetime64 array
    np_times = np.array(orbit_times, dtype='datetime64[m]')
    return np_times


def print_timestamp_satellite_info_from_global_attrs(orbit_time_stamp):
    timestamps_raw = orbit_time_stamp.strip().split()
    # Parse each timestamp
    orbit_times = []
    for timestamp in timestamps_raw:
        year = int(timestamp[0:4])
        julian_day = int(timestamp[4:7])
        hour = int(timestamp[7:9])
        minute = int(timestamp[9:11])
        if timestamp[11] == 'T':
            satellite = 'Terra'
        else:
            satellite = 'Aqua'

        dt_obj = dt(year, 1, 1) + timedelta(days=julian_day - 1, hours=hour, minutes=minute)
        orbit_times.append({'datetime': dt_obj, 'satellite': satellite})

    for i, orbit_info in enumerate(orbit_times):
        print(f"Orbit {i}: {orbit_info['datetime']} - Satellite: {orbit_info['satellite']}")



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
        'end': '2018-12-31',
        'tile': 'h09v04'
    },
    'OSBS': {
        'coords': (-81.993431, 29.689282),
        'start': '2021-01-01',
        'end': '2022-12-31',
        'tile': 'h10v06'
    },
    'Ca-Obs': {
        'coords': (-105.1178, 53.9872),
        'start': '2018-01-01',
        'end': '2022-12-31',
        'tile': 'h11v03'
    },
    'DEJU': {
        'coords': (-145.75136, 63.88112),
        'start': '2019-01-01',
        'end': '2021-12-31',
        'tile': 'h11v02'
    },
    'US-UMB': {
        'coords': (-84.7138, 45.5598),
        'start': '2018-01-01',
        'end': '2019-12-31',
        'tile': 'h12v04'
    },
    'US-Ne3': {
        'coords': (-96.4397, 41.1797),
        'start': '2018-01-01',
        'end': '2019-12-31',
        'tile': 'h10v04'
    }
}

def clip_2dcurv_ds_to_point(ds, lon_pt, lat_pt):
    # Find the x, y indices where lon_pt and lat_pt are found in ds
    lon_2d = ds.lon.values
    lat_2d = ds.lat.values

    # Compute the absolute difference and find the indices of the minimum
    dist = np.abs(lon_2d - lon_pt) + np.abs(lat_2d - lat_pt)
    y_idx, x_idx = np.unravel_index(np.argmin(dist), lon_2d.shape)
    lon_selected = ds.lon.isel(x=x_idx, y=y_idx).values
    lat_selected = ds.lat.isel(x=x_idx, y=y_idx).values
    print(f"Nearest point in dataset to ({lon_pt}, {lat_pt}) is at ({lon_selected}, {lat_selected}) with distance {dist[y_idx, x_idx]:.6f}")
    ds_point = ds.isel(x=x_idx, y=y_idx)
    return ds_point
#########################################
# Begin main
#########################################
#%%
def main():

    # mark start time to keep track of elapsed
    start_total = time.time()
    try:

        site_name = sys.argv[1]  # get the site name from the command line arguments
        
        MODIS_tile = site_info[site_name]['tile']
        site_lon = site_info[site_name]['coords'][0]
        site_lat = site_info[site_name]['coords'][1]
        composite_years = np.arange(int(site_info[site_name]['start'][:4]), int(site_info[site_name]['end'][:4]) + 1)



        QC_descr = sys.argv[2]

        MCD19_dir_QC = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/pre-fill_OutlierFilter/QCfilt_{QC_descr}/{MODIS_tile}/'

        out_basedir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/data/QC_PhotoSpec_sites_backup/CV-MVC/pre-fill_OutlierFilter/'
        os.makedirs(out_basedir, exist_ok=True)


        ### Load all the MCD19 data



        MCD19_QC_arr = []



        for composite_year in composite_years:
            print(f"Loading MCD19 year {composite_year}...")

            
            for composite_nn in range(0, 23):
                print(f"  composite {composite_nn:02d}...")

                # List all files for this month portion
                # e.g. MCD19A1.h09v05.061_1km_16day_CV-MVC_QCfilt_CloudFree_AllAOD_AllAdj_2025-21.nc
                MCD19A1_QC_files = glob.glob(f'{MCD19_dir_QC}MCD19A1.{MODIS_tile}.*_{composite_year}-{composite_nn:02d}.nc')

                if len(MCD19A1_QC_files) > 0:
                    MCD19A1_QC_file = MCD19A1_QC_files[0]  # Assuming there's only one file per month portion
                    MCD19A1_QC_ds_curv = xr.open_dataset(MCD19A1_QC_file)
                    MCD19A1_QC_ds_pt = clip_2dcurv_ds_to_point(MCD19A1_QC_ds_curv, site_lon, site_lat)
                    MCD19_QC_arr.append(MCD19A1_QC_ds_pt)
                else:
                    print(f"No files for {composite_year}-{composite_nn:02d} in {MCD19_dir_QC}")
        
        if len(MCD19_QC_arr) > 0:
            # Compile arrays and compute VIs
            MCD19_QC_pt = xr.concat(MCD19_QC_arr, dim='time')
            MCD19_QC_pt = MCD19_QC_pt.assign(CCI = (MCD19_QC_pt.Sur_refl11 - MCD19_QC_pt.Sur_refl1) / (MCD19_QC_pt.Sur_refl11 + MCD19_QC_pt.Sur_refl1))
            MCD19_QC_pt = MCD19_QC_pt.assign(NDVI = (MCD19_QC_pt.Sur_refl2 - MCD19_QC_pt.Sur_refl1) / (MCD19_QC_pt.Sur_refl2 + MCD19_QC_pt.Sur_refl1))
            
            ### Save data
            MCD19_QC_temp_file = os.path.join(out_basedir, f'QCfilt_{QC_descr}/MCD19_QCfilt_{QC_descr}_{site_name}.nc')
            Path(os.path.dirname(MCD19_QC_temp_file)).mkdir(parents=True, exist_ok=True)
            MCD19_QC_pt.to_netcdf(MCD19_QC_temp_file)
            print(f"Saved MCD19 QC data to {MCD19_QC_temp_file}")

        else:
            print(f"No MCD19 QC data found for site {site_name} in years {composite_years}. Exiting.")


    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)        

if __name__ == '__main__':
   main()