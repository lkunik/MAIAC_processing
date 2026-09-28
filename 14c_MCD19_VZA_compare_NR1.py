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
site_name = 'US-NR1'

MODIS_tile = 'h09v05'

MCD19_dir_CloudFree = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/lm-fill/QCfilt_CloudFree/{MODIS_tile}/'
MCD19_dir_basicfill = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/basic-fill/QCfilt_CloudFree/{MODIS_tile}/'


Photospec_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/data/PhotoSpec'
Photospec_file = os.path.join(Photospec_dir, f'PhotoSpec_{site_name}.csv')


MOD13_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/MODIS/NDVI_EVI_MOD13A2/01d'
MCD43_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/MODIS/Reflectance_MCD43A4/01d'



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

### Load all the MCD19 data

MCD19_CloudFree_arr = []
MCD19_basicfill_arr = []
MOD13_arr = []
MCD43_arr = []

composite_years = np.arange(2000, 2026)

#%%
# # composite_year = 2015
# for composite_year in composite_years:
#     print(f"Loading MCD19 year {composite_year}...")

#     # composite_nn = 12
#     for composite_nn in range(0, 23):
#         print(f"  composite {composite_nn:02d}...")

#         # List all files for this month portion

#         MCD19A1_CloudFree_file = os.path.join(MCD19_dir_CloudFree, f'MCD19A1.{MODIS_tile}.061_1km_16day_CV-MVC_lmFill_QCfilt_CloudFree_{composite_year}-{composite_nn:02d}.nc')
#         MCD19A1_basicfill_file = os.path.join(MCD19_dir_basicfill, f'MCD19A1.{MODIS_tile}.061_1km_16day_CV-MVC_basic-fill_QCfilt_CloudFree_{composite_year}-{composite_nn:02d}.nc')

#         if not os.path.exists(MCD19A1_CloudFree_file):
#             print(f"File does not exist: {MCD19A1_CloudFree_file}")
#             continue
#         MCD19A1_ds_curv = xr.open_dataset(MCD19A1_CloudFree_file)
#         MCD19A1_ds_pt = clip_2dcurv_ds_to_point(MCD19A1_ds_curv, site_lon, site_lat)
#         MCD19_CloudFree_arr.append(MCD19A1_ds_pt)

        
#         if not os.path.exists(MCD19A1_basicfill_file):
#             print(f"File does not exist: {MCD19A1_basicfill_file}")
#             continue
#         MCD19A1_ds_curv = xr.open_dataset(MCD19A1_basicfill_file)
#         MCD19A1_ds_pt = clip_2dcurv_ds_to_point(MCD19A1_ds_curv, site_lon, site_lat)
#         MCD19_basicfill_arr.append(MCD19A1_ds_pt)

        
    
#     print(f"Loading MOD13 year {composite_year}...")
#     MOD13_file = os.path.join(MOD13_dir, f'MOD13A2.061_01d_aid0001_WUS_{composite_year}.nc')
#     MOD13_xr = xr.open_dataset(MOD13_file, decode_coords = 'all').sortby(['x', 'y'], ascending = True)
#     MOD13_xr.rio.write_crs('EPSG:4326', inplace=True)
#     MOD13_xr = MOD13_xr.rename({f'_1_km_16_days_NDVI':'NDVI'})

#     MOD13_xr_site = MOD13_xr.sel(x=site_lon, y=site_lat, method='nearest')
#     MOD13_arr.append(MOD13_xr_site)


#     print(f"Loading MCD43 year {composite_year}...")
#     MCD43_file = os.path.join(MCD43_dir, f'MCD43A4.061_01d_8day_aid0001_WUS_{composite_year}.nc')
#     MCD43_xr = xr.open_dataset(MCD43_file, decode_coords = 'all').sortby(['lon', 'lat'], ascending = True)
#     MCD43_xr.rio.write_crs('EPSG:4326', inplace=True)
#     MCD43_xr_site = MCD43_xr.sel(lon=site_lon, lat=site_lat, method='nearest')
#     MCD43_arr.append(MCD43_xr_site)

# MCD19_CloudFree_pt = xr.concat(MCD19_CloudFree_arr, dim='time')
# MCD19_basicfill_pt = xr.concat(MCD19_basicfill_arr, dim='time')
# MOD13_pt = xr.concat(MOD13_arr, dim='time')
# MCD43_pt = xr.concat(MCD43_arr, dim='time')

# # # remove first time element - delete this ! it was fixing a weird testbench issue
# # MCD19_CloudFree_pt = MCD19_CloudFree_pt.isel(time=slice(1, None))
# # MCD19_basicfill_pt = MCD19_basicfill_pt.isel(time=slice(1, None))

# MOD13_pt['time'] = pd.to_datetime([pd.Timestamp(t.isoformat()) if hasattr(t, 'isoformat') else t for t in MOD13_pt.time.values])
# MCD43_pt['time'] = pd.to_datetime(MCD43_pt.time.values)

# ## Compute CCI from MCD19
# MCD19_CloudFree_pt = MCD19_CloudFree_pt.assign(CCI = (MCD19_CloudFree_pt.Sur_refl11 - MCD19_CloudFree_pt.Sur_refl1) / (MCD19_CloudFree_pt.Sur_refl11 + MCD19_CloudFree_pt.Sur_refl1))
# MCD19_basicfill_pt = MCD19_basicfill_pt.assign(CCI = (MCD19_basicfill_pt.Sur_refl11 - MCD19_basicfill_pt.Sur_refl1) / (MCD19_basicfill_pt.Sur_refl11 + MCD19_basicfill_pt.Sur_refl1))

# ## Compute NDVI from MCD19
# MCD19_CloudFree_pt = MCD19_CloudFree_pt.assign(NDVI = (MCD19_CloudFree_pt.Sur_refl2 - MCD19_CloudFree_pt.Sur_refl1) / (MCD19_CloudFree_pt.Sur_refl2 + MCD19_CloudFree_pt.Sur_refl1))
# MCD19_basicfill_pt = MCD19_basicfill_pt.assign(NDVI = (MCD19_basicfill_pt.Sur_refl2 - MCD19_basicfill_pt.Sur_refl1) / (MCD19_basicfill_pt.Sur_refl2 + MCD19_basicfill_pt.Sur_refl1))


# #%%
# # save data to temp
# temp_dat_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/data/temp'
# MCD19_CloudFree_temp_file = os.path.join(temp_dat_dir, f'MCD19_CloudFree_pt_{site_name}.nc')
# MCD19_basicfill_temp_file = os.path.join(temp_dat_dir, f'MCD19_basicfill_pt_{site_name}.nc')
# MOD13_temp_file = os.path.join(temp_dat_dir, f'MOD13_pt_{site_name}.nc')
# MCD43_temp_file = os.path.join(temp_dat_dir, f'MCD43_pt_{site_name}.nc')
# MCD19_CloudFree_pt.to_netcdf(MCD19_CloudFree_temp_file)
# MCD19_basicfill_pt.to_netcdf(MCD19_basicfill_temp_file)
# MOD13_pt.to_netcdf(MOD13_temp_file)
# MCD43_pt.to_netcdf(MCD43_temp_file)


#%%
### If already saved, just load it! 


temp_dat_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/data/temp'
MCD19_CloudFree_temp_file = os.path.join(temp_dat_dir, f'MCD19_CloudFree_pt_{site_name}.nc')
MCD19_basicfill_temp_file = os.path.join(temp_dat_dir, f'MCD19_basicfill_pt_{site_name}.nc')
MOD13_temp_file = os.path.join(temp_dat_dir, f'MOD13_pt_{site_name}.nc')
MCD43_temp_file = os.path.join(temp_dat_dir, f'MCD43_pt_{site_name}.nc')
MCD19_CloudFree_pt = xr.open_dataset(MCD19_CloudFree_temp_file)
MCD19_basicfill_pt = xr.open_dataset(MCD19_basicfill_temp_file)
MOD13_pt = xr.open_dataset(MOD13_temp_file)
MCD43_pt = xr.open_dataset(MCD43_temp_file)

MCD19_CloudFree_pt['VZA'] = rad_to_deg(arccos_v(MCD19_CloudFree_pt['cosVZA']))
np.unique(MCD19_CloudFree_pt['cosVZA'].values)
#%%
# Plot histogram of MCD19_CloudFree_pt['cosVZA'] values

plt.figure(figsize=(8, 4))
plt.hist(MCD19_CloudFree_pt['cosVZA'].values.flatten(), bins=50, edgecolor='black', alpha=0.7, color='#CF00BD')
plt.xlabel('cos(VZA)')
plt.ylabel('Frequency')
plt.title('Distribution of cos(VZA) in MCD19_CloudFree_pt')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#%%
# Plot seasonality (Jan 1 - Dec 31) for each year in the MCD19 CloudFree and BasicFill point series

years = np.unique(pd.to_datetime(MOD13_pt.time.values).year)

for year in years:
    # select data for the calendar year (robust to non-monotonic time index)
    start = np.datetime64(f'{year}-01-01')
    end = np.datetime64(f'{year}-12-31')

    if getattr(MCD19_CloudFree_pt, 'time', None) is not None and MCD19_CloudFree_pt.time.size > 0:
        cf_mask = (MCD19_CloudFree_pt.time.values >= start) & (MCD19_CloudFree_pt.time.values <= end)
        cf_sel = MCD19_CloudFree_pt.isel(time=cf_mask)
    else:
        cf_sel = MCD19_CloudFree_pt

    mod13_sel = MOD13_pt.sel(time=slice(start, end))

    if (getattr(cf_sel, 'time', None) is None or cf_sel.time.size == 0) and (getattr(mod13_sel, 'time', None) is None or mod13_sel.time.size == 0):
        continue

    doy_cf = pd.to_datetime(cf_sel.time.values).dayofyear if getattr(cf_sel, 'time', None) is not None and cf_sel.time.size>0 else np.array([])
    doy_bf = pd.to_datetime(mod13_sel.time.values).dayofyear if getattr(mod13_sel, 'time', None) is not None and mod13_sel.time.size>0 else np.array([])

    fig, ax = plt.subplots(figsize=(8, 3))
    if doy_cf.size > 0:
        ax.plot(doy_cf, cf_sel['NDVI'].values, marker='o', linestyle='-', label='MCD19', color='#CF00BD')
    if doy_bf.size > 0:
        ax.plot(doy_bf, mod13_sel['NDVI'].values, marker='s', linestyle='-', label='MOD13', color='#50963F')

    ax.set_xlim(1, 366)
    month_starts = [pd.Timestamp(f'{year}-{m:02d}-01').dayofyear for m in range(1, 13)]
    month_labels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
    ax.set_xticks(month_starts)
    ax.set_xticklabels(month_labels)
    ax.set_xlabel('Day of Year')
    ax.set_ylabel('NDVI')
    ax.set_title(f'{site_name} - Seasonality {year}')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



#%%

# Plot seasonality (Jan 1 - Dec 31) for each year in the MCD19 CloudFree and BasicFill point series

years = np.unique(pd.to_datetime(MOD13_pt.time.values).year)

for year in years:
    # select data for the calendar year (robust to non-monotonic time index)
    start = np.datetime64(f'{year}-01-01')
    end = np.datetime64(f'{year}-12-31')

    if getattr(MCD19_CloudFree_pt, 'time', None) is not None and MCD19_CloudFree_pt.time.size > 0:
        cf_mask = (MCD19_CloudFree_pt.time.values >= start) & (MCD19_CloudFree_pt.time.values <= end)
        cf_sel = MCD19_CloudFree_pt.isel(time=cf_mask)
    else:
        cf_sel = MCD19_CloudFree_pt

    mod13_sel = MOD13_pt.sel(time=slice(start, end))

    if (getattr(cf_sel, 'time', None) is None or cf_sel.time.size == 0) and (getattr(mod13_sel, 'time', None) is None or mod13_sel.time.size == 0):
        continue

    doy_cf = pd.to_datetime(cf_sel.time.values).dayofyear if getattr(cf_sel, 'time', None) is not None and cf_sel.time.size>0 else np.array([])
    doy_bf = pd.to_datetime(mod13_sel.time.values).dayofyear if getattr(mod13_sel, 'time', None) is not None and mod13_sel.time.size>0 else np.array([])

    fig, axes = plt.subplots(2, 1, figsize=(8, 5))
    month_starts = [pd.Timestamp(f'{year}-{m:02d}-01').dayofyear for m in range(1, 13)]
    month_labels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
    
    # Top panel: NDVI comparison
    ax = axes[0]
    if doy_cf.size > 0:
        ax.plot(doy_cf, cf_sel['NDVI'].values, marker='o', linestyle='-', label='MCD19', color='#CF00BD')
    if doy_bf.size > 0:
        ax.plot(doy_bf, mod13_sel['NDVI'].values, marker='s', linestyle='-', label='MOD13', color='#50963F')

    ax.set_xlim(1, 366)
    ax.set_xticks(month_starts)
    ax.set_xticklabels(month_labels)
    ax.set_ylabel('NDVI')
    ax.set_title(f'{site_name} - Seasonality {year}')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Bottom panel: VZA comparison
    ax = axes[1]
    if doy_cf.size > 0:
        ax.plot(doy_cf, cf_sel['VZA'].values, marker='o', linestyle='-', label='MCD19', color='#CF00BD')
    if doy_bf.size > 0:
        ax.plot(doy_bf, mod13_sel['_1_km_16_days_view_zenith_angle'].values, marker='s', linestyle='-', label='MOD13', color='#50963F')

    ax.set_xlim(1, 366)
    ax.set_xticks(month_starts)
    ax.set_xticklabels(month_labels)
    ax.set_xlabel('Day of Year')
    ax.set_ylabel('View Zenith Angle (degrees)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()



