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



#%%
# Plot time series of CCI from MCD19_CloudFree_pt and MCD19_basicfill_pt
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(MCD19_CloudFree_pt.time.values, MCD19_CloudFree_pt['CCI'].values, 
        marker='o', linestyle='-', label='MCD19 CloudFree', linewidth=2, markersize=4, alpha=0.7)
ax.plot(MCD19_basicfill_pt.time.values, MCD19_basicfill_pt['CCI'].values, 
        marker='s', linestyle='-', label='MCD19 Basic Fill', linewidth=2, markersize=4, alpha=0.7)
ax.set_xlabel('Time')
ax.set_ylabel('CCI')
ax.set_title(f'{site_name} - CCI Time Series Comparison')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(DateFormatter('%Y'))
fig.autofmt_xdate(rotation=45)
plt.tight_layout()
plt.show()

#%%
# Plot seasonality (Jan 1 - Dec 31) for each year in the MCD19 CloudFree and BasicFill point series
try:
    # gather years present in either dataset
    years_cf = pd.to_datetime(MCD19_CloudFree_pt.time.values).year
    years_bf = pd.to_datetime(MCD19_basicfill_pt.time.values).year
    years = np.unique(np.concatenate((years_cf, years_bf)))
except Exception:
    years = []

for year in years:
    # select data for the calendar year (robust to non-monotonic time index)
    start = np.datetime64(f'{year}-01-01')
    end = np.datetime64(f'{year}-12-31')

    if getattr(MCD19_CloudFree_pt, 'time', None) is not None and MCD19_CloudFree_pt.time.size > 0:
        cf_mask = (MCD19_CloudFree_pt.time.values >= start) & (MCD19_CloudFree_pt.time.values <= end)
        cf_sel = MCD19_CloudFree_pt.isel(time=cf_mask)
    else:
        cf_sel = MCD19_CloudFree_pt

    if getattr(MCD19_basicfill_pt, 'time', None) is not None and MCD19_basicfill_pt.time.size > 0:
        bf_mask = (MCD19_basicfill_pt.time.values >= start) & (MCD19_basicfill_pt.time.values <= end)
        bf_sel = MCD19_basicfill_pt.isel(time=bf_mask)
    else:
        bf_sel = MCD19_basicfill_pt

    if (getattr(cf_sel, 'time', None) is None or cf_sel.time.size == 0) and (getattr(bf_sel, 'time', None) is None or bf_sel.time.size == 0):
        continue

    doy_cf = pd.to_datetime(cf_sel.time.values).dayofyear if getattr(cf_sel, 'time', None) is not None and cf_sel.time.size>0 else np.array([])
    doy_bf = pd.to_datetime(bf_sel.time.values).dayofyear if getattr(bf_sel, 'time', None) is not None and bf_sel.time.size>0 else np.array([])

    fig, ax = plt.subplots(figsize=(8, 3))
    if doy_cf.size > 0:
        ax.plot(doy_cf, cf_sel['CCI'].values, marker='o', linestyle='-', label='MCD19 CloudFree', color='#CF00BD')
    if doy_bf.size > 0:
        ax.plot(doy_bf, bf_sel['CCI'].values, marker='s', linestyle='-', label='MCD19 BasicFill', color='#08A0FF')

    ax.set_xlim(1, 366)
    month_starts = [pd.Timestamp(f'{year}-{m:02d}-01').dayofyear for m in range(1, 13)]
    month_labels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
    ax.set_xticks(month_starts)
    ax.set_xticklabels(month_labels)
    ax.set_xlabel('Day of Year')
    ax.set_ylabel('CCI')
    ax.set_title(f'{site_name} - Seasonality {year}')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

#%%


years = np.unique(pd.to_datetime(MCD19_CloudFree_pt.time.values).year)

for year in years:
    # select data for the calendar year (robust to non-monotonic time index)
    start = np.datetime64(f'{year}-01-01')
    end = np.datetime64(f'{year}-12-31')


    if getattr(MCD19_CloudFree_pt, 'time', None) is not None and MCD19_CloudFree_pt.time.size > 0:
        cf_mask = (MCD19_CloudFree_pt.time.values >= start) & (MCD19_CloudFree_pt.time.values <= end)
        cf_sel = MCD19_CloudFree_pt.isel(time=cf_mask)
    else:
        cf_sel = MCD19_CloudFree_pt

    if getattr(MCD19_basicfill_pt, 'time', None) is not None and MCD19_basicfill_pt.time.size > 0:
        bf_mask = (MCD19_basicfill_pt.time.values >= start) & (MCD19_basicfill_pt.time.values <= end)
        bf_sel = MCD19_basicfill_pt.isel(time=bf_mask)
    else:
        bf_sel = MCD19_basicfill_pt

    mod13_sel = MOD13_pt.sel(time=slice(start, end))
    mcd43_sel = MCD43_pt.sel(time=slice(start, end))

    # Skip year if no data in any dataset
    if (cf_sel.time.size == 0 and bf_sel.time.size == 0 and 
        mod13_sel.time.size == 0 and mcd43_sel.time.size == 0):
        continue

    # Extract day of year for each dataset
    doy_cf = pd.to_datetime(cf_sel.time.values).dayofyear if cf_sel.time.size > 0 else np.array([])
    doy_bf = pd.to_datetime(bf_sel.time.values).dayofyear if bf_sel.time.size > 0 else np.array([])
    doy_mod13 = pd.to_datetime(mod13_sel.time.values).dayofyear if mod13_sel.time.size > 0 else np.array([])
    doy_mcd43 = pd.to_datetime(mcd43_sel.time.values).dayofyear if mcd43_sel.time.size > 0 else np.array([])

    fig, ax = plt.subplots(figsize=(10, 4))
    
    if doy_cf.size > 0:
        ax.plot(doy_cf, cf_sel['NDVI'].values, marker='o', linestyle='-', label='MCD19 CloudFree', color='#CF00BD')
    if doy_bf.size > 0:
        ax.plot(doy_bf, bf_sel['NDVI'].values, marker='s', linestyle='-', label='MCD19 BasicFill', color='#08A0FF')
    if doy_mod13.size > 0:
        ax.plot(doy_mod13, mod13_sel['NDVI'].values, marker='^', linestyle='-', label='MOD13', color='#35DCE8')
    if doy_mcd43.size > 0:
        ax.plot(doy_mcd43, mcd43_sel['NDVI'].values, marker='s', linestyle='-', label='MCD43', color='#E69F00')


    ax.set_xlim(1, 366)
    month_starts = [pd.Timestamp(f'{year}-{m:02d}-01').dayofyear for m in range(1, 13)]
    month_labels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
    ax.set_xticks(month_starts)
    ax.set_xticklabels(month_labels)
    ax.set_xlabel('Day of Year')
    ax.set_ylabel('NDVI')
    ax.set_title(f'{site_name} - NDVI Seasonality {year}')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

#%%







#%%

Photospec_df = pd.read_csv(Photospec_file)
print("Photospec_df columns:", Photospec_df.columns.tolist())

# Remove any data from Photospec_df where NDVI is below 0.6
Photospec_df = Photospec_df[Photospec_df['NDVI'] >= 0.6].reset_index(drop=True)
# Convert Photospec timestamps from "M/D/YY HH:MM" format to datetime64
Photospec_times = pd.to_datetime(Photospec_df['Time'], format='%m/%d/%y %H:%M')

# Select MCD19 data within the time range of Photospec data
MCD19_CloudFree_pt = MCD19_CloudFree_pt.sel(time=slice(Photospec_times.iloc[0], Photospec_times.iloc[-1]))
MCD19_basicfill_pt = MCD19_basicfill_pt.sel(time=slice(Photospec_times.iloc[0], Photospec_times.iloc[-1]))

#############################################################
### Load SNODAS 0.05° for approximate snow depth filter
#############################################################

#%%
SNODAS_file = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/SNODAS/masked/05d/SNODAS_WUS_16day_05d_2003-2024.nc'
snow_depth_thresh_mm = 200  #  mm snow depth threshold

snodas_xr_raw = xr.open_dataset(SNODAS_file, decode_coords="all").sortby(['lat', 'lon'], ascending=True)
snodas_xr = snodas_xr_raw['SNWZ'].sel(lon = site_lon, lat = site_lat, method="nearest") # Get snow depth variable
snodas_xr = snodas_xr.sel(time=slice('2017-06-01', '2018-06-30'))
snodas_xr_MODtime = snodas_xr.interp(time=MCD19_CloudFree_pt.time, method="linear")

# Find the earliest timestamp where SNODAS snow depth exceeds the threshold
snow_exceed_mask = snodas_xr > snow_depth_thresh_mm
snow_exceed_start = snodas_xr.time.values[snow_exceed_mask].min()
snow_exceed_end = snodas_xr.time.values[snow_exceed_mask].max()
# Convert SNODAS time to day of year for plotting
snodas_doy = pd.to_datetime(snodas_xr.time.values).dayofyear

plt.figure(figsize=(7, 2.5))
plt.axhline(y=snow_depth_thresh_mm, color='red', linestyle='--', label=f'Snow Depth Threshold: {snow_depth_thresh_mm} mm')
plt.scatter(snodas_doy, snodas_xr.values, color='blue', label='SNODAS Snow Depth', s=20, alpha=0.8)
plt.xlabel('Day of Year')
plt.ylabel('Snow Depth (mm)')
plt.title('SNODAS Snow Depth vs Day of Year')
plt.tight_layout()
plt.show()


# Print dayofyear values where snow depth exceeds threshold
first_highsnow_doy = np.nanmin(snodas_doy[snodas_xr.values > snow_depth_thresh_mm])
last_highsnow_doy = np.nanmax(snodas_doy[snodas_xr.values > snow_depth_thresh_mm])
print(f"First day of year with SNODAS snow depth > {snow_depth_thresh_mm} mm: {first_highsnow_doy}")
print(f"Last day of year with SNODAS snow depth > {snow_depth_thresh_mm} mm: {last_highsnow_doy}")





# # Resample MCD43_df to 8-day resolution
# MCD43_df['date'] = MCD43_times
# MCD43_df_8day = MCD43_pt.set_index('date').resample('8D').mean(numeric_only=True).reset_index()
# MCD43_times_8day = pd.to_datetime(MCD43_df_8day['date'])


# Filter Photospec_df for times between 13:00 and 14:00
mask = (Photospec_times.dt.hour >= 13) & (Photospec_times.dt.hour < 14)
Photospec_df_filtered = Photospec_df[mask].reset_index(drop=True)
Photospec_times_filtered = Photospec_times[mask].reset_index(drop=True)

# Set x-axis to month abbreviations at the start of each month
months = np.arange(1, 13)
month_starts = [pd.Timestamp(f'{composite_years[0]}-{m:02d}-01').dayofyear for m in months]
month_labels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']




# %%

MCD19_CloudFree_pt_PStime = MCD19_CloudFree_pt.sel(time=slice(Photospec_times_filtered.iloc[0], Photospec_times_filtered.iloc[-1]))
MCD19_basicfill_pt_PStime = MCD19_basicfill_pt.sel(time=slice(Photospec_times_filtered.iloc[0], Photospec_times_filtered.iloc[-1]))
MOD13_pt_PStime = MOD13_pt.sel(time=slice(Photospec_times_filtered.iloc[0], Photospec_times_filtered.iloc[-1]))
MCD43_pt_PStime = MCD43_pt.sel(time=slice(Photospec_times_filtered.iloc[0], Photospec_times_filtered.iloc[-1]))

MCD19_CloudFree_times = np.array([np.datetime64(t) for t in MCD19_CloudFree_pt_PStime['time'].values])
MCD19_basicfill_times = np.array([np.datetime64(t) for t in MCD19_basicfill_pt_PStime['time'].values])
MOD13_times = np.array([np.datetime64(t) for t in MOD13_pt_PStime['time'].values])
MCD43_times = np.array([np.datetime64(t) for t in MCD43_pt_PStime['time'].values])



# Calculate day of year for MCD19_CloudFree_times and Photospec_times_filtered
mcd19_doy = pd.to_datetime(MCD19_CloudFree_times).dayofyear
mcd19_basicfill_doy = pd.to_datetime(MCD19_basicfill_times).dayofyear
mod13_doy = pd.to_datetime(MOD13_times).dayofyear
mcd43_doy = pd.to_datetime(MCD43_times).dayofyear
photospec_doy = Photospec_times_filtered.dt.dayofyear

# Sort data by day of year so you can plot as lines without weird zigzags
mcd19_doy_sorted = mcd19_doy[np.argsort(mcd19_doy)]
mcd19_ndvi_sorted = MCD19_CloudFree_pt_PStime['NDVI'].values[np.argsort(mcd19_doy)]
mcd19_basicfill_doy_sorted = mcd19_basicfill_doy[np.argsort(mcd19_basicfill_doy)]
mcd19_basicfill_ndvi_sorted = MCD19_basicfill_pt_PStime['NDVI'].values[np.argsort(mcd19_basicfill_doy)]

mod13_doy_sorted = mod13_doy[np.argsort(mod13_doy)]
mod13_ndvi_sorted = MOD13_pt_PStime['NDVI'].values[np.argsort(mod13_doy)]
mcd43_doy_sorted = mcd43_doy[np.argsort(mcd43_doy)]
mcd43_ndvi_sorted = MCD43_pt_PStime['NDVI'].values[np.argsort(mcd43_doy)]

#%%
# Set global matplotlib font sizes for axes and legends
plt.rcParams.update({
   'axes.labelsize': 15,
   'axes.titlesize': 15,
   'xtick.labelsize': 15,
   'ytick.labelsize': 15,
   'legend.fontsize': 13
})

MOD13_color = '#35DCE8'
MCD19_color = '#CF00BD'
MCD19_basicfill_color = '#08A0FF'
MCD43_color = '#E69F00'

fig, ax = plt.subplots(figsize=(7, 3))
# Plot shaded region where snow depth exceeds threshold
# ax.axvspan(first_highsnow_doy - 15, last_highsnow_doy + 15, color='lightblue', label=f'SNODAS snow depth > {snow_depth_thresh_mm} mm', alpha=0.5)

# Plot Photospec NDVI on left y-axis
ax.scatter(photospec_doy, Photospec_df_filtered['NDVI'], label='Photospec', color='black', s=10, alpha=0.7)
ax.scatter(mod13_doy_sorted, mod13_ndvi_sorted, label='MOD13 NDVI', color=MOD13_color, s=20, alpha=1)
ax.scatter(mcd19_doy_sorted, mcd19_ndvi_sorted, label='MCD19 NDVI', color=MCD19_color, s=20, alpha=1)
ax.scatter(mcd19_basicfill_doy_sorted, mcd19_basicfill_ndvi_sorted, label='MCD19 Basic Fill', color=MCD19_basicfill_color, s=20, alpha=1)
ax.scatter(mcd43_doy_sorted, mcd43_ndvi_sorted, label='MCD43 NDVI', color=MCD43_color, s=20, alpha=1)


# Adjust limits and legend
ax.set_ylabel('NDVI')
ax.set_xticks(month_starts)
ax.set_xticklabels(month_labels)


# Combine legends using Patch
handles = [
    Patch(color='lightblue', label=f'SNODAS snow\ndepth > {snow_depth_thresh_mm} mm', alpha=0.5),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label='Photospec'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=MCD19_color, markersize=8, label='MCD19'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=MCD19_basicfill_color, markersize=8, label='MCD19 Basic Fill'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=MOD13_color, markersize=8, label='MOD13'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=MCD43_color, markersize=8, label='MCD43')
]
ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.05, 1))
ax.set_title(site_name)
plt.tight_layout()
plt.show()

#%%

fig, ax = plt.subplots(figsize=(6, 3))

# # Plot shaded region where snow depth exceeds threshold
ax.axvspan(snow_exceed_start, snow_exceed_end, color='lightblue', label=f'Snow cover', alpha=0.3)

# Plot Photospec NDVI on left y-axis
ax.scatter(Photospec_times_filtered, Photospec_df_filtered['NDVI'], label='Photospec', color='black', s=10, alpha=0.7)
ax.scatter(MOD13_times, MOD13_pt_PStime['NDVI'], label='MOD13 NDVI', color=MOD13_color, s=28, alpha=1)
ax.scatter(MCD19_CloudFree_times, MCD19_CloudFree_pt_PStime['NDVI'].values, label='MCD19', color=MCD19_color, s=28, alpha=1)
ax.scatter(MCD19_basicfill_times, MCD19_basicfill_pt_PStime['NDVI'].values, label='MCD19 Basic Fill', color=MCD19_basicfill_color, s=28, alpha=1)
ax.scatter(MCD43_times, MCD43_pt_PStime['NDVI'], label='MCD43 NDVI', color=MCD43_color, s=28, alpha=1)

# Adjust limits and legend
ax.set_ylabel('NDVI')
ax.yaxis.set_major_formatter(FFmt(lambda y, _: f'{y:.1f}'))

ax.set_xlim(Photospec_times_filtered.min() - pd.Timedelta(days=10), Photospec_times_filtered.max() + pd.Timedelta(days=10))
ax.xaxis.set_major_locator(MonthLocator(bymonth=[7, 10], interval=1))
ax.xaxis.set_major_locator(MonthLocator(bymonth=[1, 4, 7, 10]))
ax.xaxis.set_major_formatter(DateFormatter('%b\n%Y'))
# years = pd.date_range(start=Photospec_times_filtered.min(), end=Photospec_times_filtered.max(), freq='YS')
# ax.set_xticks(years)# Set x-axis ticks to show years only
# ax.set_xticklabels([str(year.year) for year in years])
# Combine legends using Patch
# handles = [
#     Patch(color='lightblue', label=f'Snow cover', alpha=0.3),
#     plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label='Photospec'),
#     plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=MCD19_color, markersize=8, label='MCD19'),
#     plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=MOD13_color, markersize=8, label='MOD13'),
#     plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=MCD43_color, markersize=8, label='MCD43')
# ]
# ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.05, 1))
ax.set_title(f'{site_name} (lat: {site_lat:.2f}, lon: {site_lon:.2f})')
plt.tight_layout()
plt.show()

#%%


# Create a figure with two subplots (one column, two rows)
fig, axes = plt.subplots(2, 1, figsize=(7.5, 4.7), sharex=True)

# --- Top subplot: NDVI comparison ---
ax = axes[0]
ax.axvspan(first_highsnow_doy - 15, last_highsnow_doy + 15, color='lightblue', label=f'SNODAS snow depth > {snow_depth_thresh_mm} mm', alpha=0.5)
ax.scatter(photospec_doy, Photospec_df_filtered['NDVI'], label='Photospec', color='black', s=10, alpha=0.7)
ax.scatter(mod13_doy_sorted, mod13_ndvi_sorted, label='MOD13 NDVI', color=MOD13_color, s=28, alpha=1)
ax.scatter(mcd19_doy_sorted, mcd19_ndvi_sorted, label='MCD19 NDVI', color=MCD19_color, s=28, alpha=1)
ax.scatter(mcd43_doy_sorted, mcd43_ndvi_sorted, label='MCD43 NDVI', color=MCD43_color, s=28, alpha=1)
ax.set_ylabel('NDVI')
ax.set_xticks(month_starts)
ax.set_xticklabels(month_labels)
ax.set_title(site_name)
handles = [
    Patch(color='lightblue', label=f'SNODAS snow\ndepth > {snow_depth_thresh_mm} mm', alpha=0.5),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label='Photospec'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=MCD19_color, markersize=8, label='MCD19'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=MOD13_color, markersize=8, label='MOD13'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=MCD43_color, markersize=8, label='MCD43')
]
ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.05, 1))

# --- Bottom subplot: CCI comparison ---
ax2 = axes[1]
ax2.axvspan(first_highsnow_doy - 15, last_highsnow_doy + 15, color='lightblue', alpha=0.5)
ax2.scatter(photospec_doy, Photospec_df_filtered['CCI'], label='Photospec CCI', color='black', s=10, alpha=0.7)
ax2.scatter(mcd19_doy_sorted, MCD19_basicfill_pt_PStime['CCI'].values[np.argsort(mcd19_doy)], label='MCD19 CCI', color=MCD19_color, s=28, alpha=1)

ax2.scatter(mcd19_doy_sorted, MCD19_basicfill_pt_PStime['CCI'].values[np.argsort(mcd19_doy)], label='MCD19 CCI', color=MCD19_color, s=28, alpha=1)
ax2.set_ylabel('CCI')
ax2.set_xticks(month_starts)
ax2.set_xticklabels(month_labels)

handles2 = [
    Patch(color='lightblue', label=f'SNODAS snow\ndepth > {snow_depth_thresh_mm} mm', alpha=0.5),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label='Photospec'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=MCD19_color, markersize=8, label='MCD19')
]
ax2.legend(handles=handles2, loc='upper left', bbox_to_anchor=(1.05, 1))

plt.tight_layout()
plt.show()

# %%

# Calculate day of year for MCD19_CloudFree_times and Photospec_times_filtered
mcd19_doy = pd.to_datetime(MCD19_CloudFree_times).dayofyear
mcd19_basicfill_doy = pd.to_datetime(MCD19_basicfill_times).dayofyear
photospec_doy = Photospec_times_filtered.dt.dayofyear

plt.figure(figsize=(9, 2.5))
ax = plt.gca()
ax2 = ax.twinx()

# Plot shaded region where snow depth exceeds threshold
ax.axvspan(first_highsnow_doy - 15, last_highsnow_doy + 15, color='lightblue', label=f'SNODAS snow depth > {snow_depth_thresh_mm} mm', alpha=0.3)

# Plot Photospec CCI on left y-axis
ax.scatter(photospec_doy, Photospec_df_filtered['CCI'], label='Photospec', color='black', s=10, alpha=0.7)
ax.set_ylabel('Photospec CCI')

# Plot MCD19 CCI on right y-axis
ax2.scatter(mcd19_doy, MCD19_CloudFree_pt_PStime['CCI'].values, label='MCD19 CCI\n(CloudFree)', color=MCD19_color, s=28, alpha=1)
ax2.set_ylabel('MCD19 CCI')

# Plot MCD19 CCI on right y-axis
ax2.scatter(mcd19_doy, MCD19_basicfill_pt_PStime['CCI'].values, label='MCD19 CCI\n(BasicFill)', color=MCD19_basicfill_color, s=28, alpha=1)
ax2.set_ylabel('MCD19 CCI')


# Adjust limits and legend
ax.set_xticks(month_starts)
ax.set_xticklabels(month_labels)


# Combine legends using Patch
handles = [
    Patch(color = 'lightblue', label=f'SNODAS snow\ndepth > {snow_depth_thresh_mm} mm', alpha=0.3),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label='PhotoSpec'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=MCD19_color, markersize=8, label='MCD19 ("Clear" only)'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=MCD19_basicfill_color, markersize=8, label='MCD19 (BasicFill)')
]
ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.24, 1))

plt.tight_layout()
plt.show()


#%%



# Calculate day of year for MCD19_CloudFree_times and Photospec_times_filtered
mcd19_doy = pd.to_datetime(MCD19_CloudFree_times).dayofyear
photospec_doy = Photospec_times_filtered.dt.dayofyear

fig, axes = plt.subplots(3, 1, figsize=(10.5, 7), sharex=True, sharey=True)
panel_labels = [
    'MCD19 ("Clear" only)',
    'MCD19 (Basic fill)',
]
cci_arrays = [
    MCD19_CloudFree_pt['CCI'].values,
    MCD19_basicfill_pt['CCI'].values
]
mcd19_cci_min = np.nanmin(cci_arrays)
mcd19_cci_max = np.nanmax(cci_arrays)
mcd19_cci_range = mcd19_cci_max - mcd19_cci_min
colors = ['red', '#08A0FF', 'orange']

for i, ax in enumerate(axes):
    ax.axvspan(first_highsnow_doy - 15, last_highsnow_doy + 15, color='lightblue', alpha=0.3)
    ax.scatter(photospec_doy, Photospec_df_filtered['CCI'], color='black', s=10, alpha=0.7, label='Photospec')
    ax2 = ax.twinx()
    ax2.scatter(mcd19_doy, cci_arrays[i], color=colors[i], s=28, alpha=1, label=panel_labels[i])
    ax.set_xticks(month_starts)
    ax.set_xticklabels(month_labels)
    
    ax.set_ylabel('Photospec CCI')
    ax2.set_ylabel("MCD19 CCI")
    ax2.set_ylim(mcd19_cci_min - (mcd19_cci_range*0.05), mcd19_cci_max + (mcd19_cci_range*0.05))
    if i == 0:
        ax.set_title(site_name)
    else:
        ax.set_title('')
    if i == 0:
        handles = [
            Patch(color='lightblue', label=f'SNODAS snow depth > {snow_depth_thresh_mm} mm', alpha=0.3),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label='PhotoSpec'),
        ]
        for j in range(len(colors)):
            handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[j], markersize=8, label=panel_labels[j]))
        ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.3, 1))

plt.tight_layout()
plt.show()

#%%



# Calculate day of year for MCD19_CloudFree_times and Photospec_times_filtered

fig, axes = plt.subplots(3, 1, figsize=(5.6, 5), sharex=True, sharey=True)
panel_labels = [
    'MCD19 ("Clear" only)',
    'MCD19 ("Clear" or\n"Possibly Cloudy")',
]
cci_arrays = [
    MCD19_CloudFree_pt['CCI'],
    MCD19_basicfill_pt['CCI'],
]
mcd19_cci_min = np.nanmin(cci_arrays)
mcd19_cci_max = np.nanmax(cci_arrays)
mcd19_cci_range = mcd19_cci_max - mcd19_cci_min
colors = ['red', '#08A0FF', 'orange']

for i, ax in enumerate(axes):
    # # Plot shaded region where snow depth exceeds threshold
    ax.axvspan(snow_exceed_start, snow_exceed_end, color='lightblue', label=f'Snow cover', alpha=0.3)

    ax.scatter(Photospec_times_filtered, Photospec_df_filtered['CCI'], color='black', s=10, alpha=0.7, label='Photospec')
    ax2 = ax.twinx()
    ax2.scatter(MCD19_CloudFree_times, cci_arrays[i], color=colors[i], s=35, alpha=1, label=panel_labels[i])

    if i == 1:
        ax.set_ylabel('Photospec CCI')
        ax2.set_ylabel("MCD19 CCI")
    else:
        ax.set_ylabel('')
        ax2.set_ylabel('')
    ax2.set_ylim(mcd19_cci_min - (mcd19_cci_range*0.05), mcd19_cci_max + (mcd19_cci_range*0.05))
    # ax.set_xlim(Photospec_times_filtered.min(), Photospec_times_filtered.max())
    ax.yaxis.set_major_formatter(FFmt(lambda y, _: f'{y:.1f}'))
    ax2.yaxis.set_major_formatter(FFmt(lambda y, _: f'{y:.1f}'))
    ax.set_xlim(Photospec_times_filtered.min() - pd.Timedelta(days=10), Photospec_times_filtered.max() + pd.Timedelta(days=10))
    ax.xaxis.set_major_locator(MonthLocator(bymonth=[7, 10], interval=1))
    ax.xaxis.set_major_locator(MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(DateFormatter('%b\n%Y'))
    # years = pd.date_range(start=Photospec_times_filtered.min(), end=Photospec_times_filtered.max(), freq='YS')
    # ax.set_xticks(years)# Set x-axis ticks to show years only
    # ax.set_xticklabels([str(year.year) for year in years])
    if i == 0:
        ax.set_title(f'{site_name} (lat: {site_lat:.2f}, lon: {site_lon:.2f})')
    else:
        ax.set_title('')
    # if i == 0:
    #     handles = [
    #         # Patch(color='lightblue', label=f'SNODAS snow depth > {snow_depth_thresh_mm} mm', alpha=0.3),
    #         plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label='PhotoSpec'),
    #     ]
    #     for j in range(len(colors)):
    #         handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[j], markersize=8, label=panel_labels[j]))
    #     ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.25, 1))

plt.tight_layout()
plt.show()

#%%
# Plot NDVI comparison between MCD19, MOD13, and MCD43

fig, axes = plt.subplots(2, 1, figsize=(5.6, 5), sharex=True, sharey=True)
panel_labels = [
    'MCD19 ("Clear" only)',
    'MCD19 ("Clear" or\n"Possibly Cloudy")',
]
ndvi_arrays = [
    MCD19_CloudFree_pt['NDVI'],
    MCD19_basicfill_pt['NDVI'],
]
mcd19_ndvi_min = np.nanmin(ndvi_arrays)
mcd19_ndvi_max = np.nanmax(ndvi_arrays)
mcd19_ndvi_range = mcd19_ndvi_max - mcd19_ndvi_min
colors_ndvi = ['red', '#08A0FF']

for i, ax in enumerate(axes):
    # Plot shaded region where snow depth exceeds threshold
    ax.axvspan(snow_exceed_start, snow_exceed_end, color='lightblue', label=f'Snow cover', alpha=0.3)

    ax.scatter(Photospec_times_filtered, Photospec_df_filtered['NDVI'], color='black', s=10, alpha=0.7, label='Photospec')
    ax2 = ax.twinx()
    ax2.scatter(MCD19_CloudFree_times, ndvi_arrays[i], color=colors_ndvi[i], s=35, alpha=1, label=panel_labels[i])
    ax2.scatter(MOD13_times, MOD13_pt['NDVI'], color=MOD13_color, s=35, alpha=0.7, marker='^', label='MOD13')
    ax2.scatter(MCD43_times, MCD43_pt['NDVI'], color=MCD43_color, s=35, alpha=0.7, marker='s', label='MCD43')

    if i == 1:
        ax.set_ylabel('Photospec NDVI')
        ax2.set_ylabel("NDVI")
    else:
        ax.set_ylabel('')
        ax2.set_ylabel('')
    ax2.set_ylim(mcd19_ndvi_min - (mcd19_ndvi_range*0.05), mcd19_ndvi_max + (mcd19_ndvi_range*0.05))
    ax.yaxis.set_major_formatter(FFmt(lambda y, _: f'{y:.1f}'))
    ax2.yaxis.set_major_formatter(FFmt(lambda y, _: f'{y:.1f}'))
    ax.set_xlim(Photospec_times_filtered.min() - pd.Timedelta(days=10), Photospec_times_filtered.max() + pd.Timedelta(days=10))
    ax.xaxis.set_major_locator(MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(DateFormatter('%b\n%Y'))
    if i == 0:
        ax.set_title(f'{site_name} NDVI Comparison (lat: {site_lat:.2f}, lon: {site_lon:.2f})')
    else:
        ax.set_title('')
    if i == 0:
        handles = [
            Patch(color='lightblue', label=f'Snow cover', alpha=0.3),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label='Photospec'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors_ndvi[0], markersize=8, label='MCD19 ("Clear")'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors_ndvi[1], markersize=8, label='MCD19 (Basic Fill)'),
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=MOD13_color, markersize=8, label='MOD13'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=MCD43_color, markersize=8, label='MCD43')
        ]
        ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.25, 1))

plt.tight_layout()
plt.show()








#%%



# Apply snow filter: set CCI to nan where snow depth exceeds threshold
MCD19_CloudFree_snowfilt = MCD19_CloudFree_pt.copy()
MCD19_CloudFree_snowfilt = MCD19_CloudFree_snowfilt.where(snodas_xr_MODtime <= snow_depth_thresh_mm)

MCD19_Cloud_snowfilt = MCD19_Cloud_pt.copy()
MCD19_Cloud_snowfilt = MCD19_Cloud_snowfilt.where(snodas_xr_MODtime <= snow_depth_thresh_mm)

MCD19_CloudAOD_snowfilt = MCD19_CloudAOD_pt.copy()
MCD19_CloudAOD_snowfilt = MCD19_CloudAOD_snowfilt.where(snodas_xr_MODtime <= snow_depth_thresh_mm)




#%%

################################################################
### Generate linear correlations between Photospec CCI and snow-filtered MCD19 CCI
################################################################

# First, Interpolate Photospec CCI to match MCD19_CloudFree_snowfilt.time
photospec_cci_interp = pd.Series(
    np.interp(
        pd.to_datetime(MCD19_CloudFree_snowfilt.time.values).astype(np.int64),
        Photospec_times_filtered.astype(np.int64),
        Photospec_df_filtered['CCI'].values
    ),
    index=MCD19_CloudFree_snowfilt.time.values
)

# Set to nan before first and after last valid Photospec time
first_time = Photospec_times_filtered.iloc[0]
last_time = Photospec_times_filtered.iloc[-1]
interp_times = pd.to_datetime(MCD19_CloudFree_snowfilt.time.values)

photospec_cci_interp[(interp_times < first_time) | (interp_times > last_time)] = np.nan

# Convert interp_times to dayofyear
interp_times_doy = pd.to_datetime(interp_times).dayofyear




# %%


# Calculate day of year for MCD19_CloudFree_times and Photospec_times_filtered
mcd19_doy = pd.to_datetime(MCD19_CloudFree_times).dayofyear
photospec_doy = Photospec_times_filtered.dt.dayofyear

fig, ax = plt.subplots(figsize=(7, 3))

# # Plot shaded region where snow depth exceeds threshold
# ax.axvspan(first_highsnow_doy - 15, last_highsnow_doy + 15, color='lightblue', label=f'SNODAS snow depth > {snow_depth_thresh_mm} mm', alpha=0.3)

# Plot Photospec CCI on left y-axis
ax.scatter(photospec_doy, Photospec_df_filtered['CCI'], label='Daily, 1-2pm', color='gray', s=14, alpha=0.5)

# Plot MCD19 CCI on right y-axis
ax.scatter(interp_times_doy, photospec_cci_interp.values, label='16-day', color='black', s=28, alpha=1, marker='^')

# Adjust limits and legend
ax.set_ylabel('Photospec CCI')
ax.set_xticks(month_starts)
ax.set_xticklabels(month_labels)

ax.set_title(site_name)
ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()



#%%


# Only compare where both are valid (not nan)
valid_mask = (~np.isnan(MCD19_CloudFree_snowfilt['CCI'].values)) & (~np.isnan(photospec_cci_interp.values))
# Compute R^2 for the scatter of interpolated Photospec CCI vs filtered MCD19 CCI
if np.sum(valid_mask) > 1:
    slope, intercept, r_value, p_value, std_err = linregress(
        photospec_cci_interp.values[valid_mask],
        MCD19_CloudFree_snowfilt['CCI'].values[valid_mask]
    )
    print(f"MCD19 CloudFree QC scheme - Fit parameters (interp): slope={slope:.3f}, intercept={intercept:.3f}, R^2={r_value**2:.3f}")
else:
    print("Not enough valid data for regression (interp).")


plt.figure(figsize=(3.5, 3.4))
plt.scatter(photospec_cci_interp.values[valid_mask], MCD19_CloudFree_snowfilt['CCI'].values[valid_mask], color='red', alpha=0.7)
# Add fit line to the scatter plot
if np.sum(valid_mask) > 1:
    x_fit = np.linspace(np.nanmin(photospec_cci_interp.values[valid_mask]), np.nanmax(photospec_cci_interp.values[valid_mask]), 100)
    y_fit = slope * x_fit + intercept
    plt.plot(x_fit, y_fit, color='black', linewidth=2, label='Fit line')
    # Add text box with slope and R^2
    textstr = f"Slope = {slope:.2f}\n$R^2$ = {r_value**2:.3f}"
    plt.gca().text(
        0.98, 0.02, textstr,
        transform=plt.gca().transAxes,
        fontsize=14,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.6)
    )
plt.ylim(mcd19_cci_min - (mcd19_cci_range*0.05), mcd19_cci_max + (mcd19_cci_range*0.05))
plt.xlabel('Photospec CCI')
plt.ylabel('MCD19 CCI')
plt.title('Clear Only')
plt.tight_layout()
plt.show()


#%%


# Only compare where both are valid (not nan)
valid_mask = (~np.isnan(MCD19_Cloud_snowfilt['CCI'].values)) & (~np.isnan(photospec_cci_interp.values))
# Compute R^2 for the scatter of interpolated Photospec CCI vs filtered MCD19 CCI
if np.sum(valid_mask) > 1:
    slope, intercept, r_value, p_value, std_err = linregress(
        photospec_cci_interp.values[valid_mask],
        MCD19_Cloud_snowfilt['CCI'].values[valid_mask]
    )
    print(f"MCD19 Cloud QC scheme - Fit parameters (interp): slope={slope:.3f}, intercept={intercept:.3f}, R^2={r_value**2:.3f}")
else:
    print("Not enough valid data for regression (interp).")


# Only compare where both are valid (not nan)
valid_mask = (~np.isnan(MCD19_Cloud_snowfilt['CCI'].values)) & (~np.isnan(photospec_cci_interp.values))
plt.figure(figsize=(3.5, 3.4))
plt.scatter(photospec_cci_interp.values[valid_mask], MCD19_Cloud_snowfilt['CCI'].values[valid_mask], color='blue', alpha=0.7, label='Filtered MCD19 vs Photospec (interp)')
# Add fit line to the scatter plot
if np.sum(valid_mask) > 1:
    x_fit = np.linspace(np.nanmin(photospec_cci_interp.values[valid_mask]), np.nanmax(photospec_cci_interp.values[valid_mask]), 100)
    y_fit = slope * x_fit + intercept
    plt.plot(x_fit, y_fit, color='black', linewidth=2, label='Fit line')
    # Add text box with slope and R^2
    textstr = f"Slope = {slope:.2f}\n$R^2$ = {r_value**2:.3f}"
    plt.gca().text(
        0.98, 0.02, textstr,
        transform=plt.gca().transAxes,
        fontsize=14,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.6)
    )
plt.ylim(mcd19_cci_min - (mcd19_cci_range*0.05), mcd19_cci_max + (mcd19_cci_range*0.05))
plt.xlabel('Photospec CCI')
plt.ylabel('MCD19 CCI')
plt.title('Clear or Possibly Cloudy')
plt.tight_layout()
plt.show()


# %%

# Only compare where both are valid (not nan)
valid_mask = (~np.isnan(MCD19_CloudAOD_snowfilt['CCI'].values)) & (~np.isnan(photospec_cci_interp.values))

# Compute R^2 for the scatter of interpolated Photospec CCI vs filtered MCD19 CCI
if np.sum(valid_mask) > 1:
    slope, intercept, r_value, p_value, std_err = linregress(
        photospec_cci_interp.values[valid_mask],
        MCD19_CloudAOD_snowfilt['CCI'].values[valid_mask]
    )
    print(f"MCD19 CloudAOD QC scheme - Fit parameters (interp): slope={slope:.3f}, intercept={intercept:.3f}, R^2={r_value**2:.3f}")
else:
    print("Not enough valid data for regression (interp).")


plt.figure(figsize=(3.5, 3.6))
plt.scatter(photospec_cci_interp.values[valid_mask], MCD19_CloudAOD_snowfilt['CCI'].values[valid_mask], color='orange', alpha=0.7, label='Filtered MCD19 vs Photospec (interp)')
if np.sum(valid_mask) > 1:
    x_fit = np.linspace(np.nanmin(photospec_cci_interp.values[valid_mask]), np.nanmax(photospec_cci_interp.values[valid_mask]), 100)
    y_fit = slope * x_fit + intercept
    plt.plot(x_fit, y_fit, color='black', linewidth=2, label='Fit line')
    # Add text box with slope and R^2
    textstr = f"Slope = {slope:.2f}\n$R^2$ = {r_value**2:.3f}"
    plt.gca().text(
        0.98, 0.02, textstr,
        transform=plt.gca().transAxes,
        fontsize=14,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.6)
    )
plt.ylim(mcd19_cci_min - (mcd19_cci_range*0.05), mcd19_cci_max + (mcd19_cci_range*0.05))
plt.xlabel('Photospec CCI')
plt.ylabel('MCD19 CCI')
plt.title('Clear or Possibly Cloudy;\nLow AOD only')
plt.tight_layout()
plt.show()



# %%
