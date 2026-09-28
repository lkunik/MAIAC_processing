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
site_name = 'US-NR1'
MODIS_tile = 'h09v04'


QC1_descr = "CloudFree_AllAOD_AllAdj"
QC2_descr = "CloudFree_AllAOD_ClearAdj"
# QC3_descr = "CloudFree_AllAOD_ClearAdj"

QC1_legend_descr = 'All Adjacency'
QC2_legend_descr = 'Clear Adjacency'

MCD19_dir_QC1 = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/pre-fill/QCfilt_{QC1_descr}/{MODIS_tile}/'
MCD19_dir_QC2 = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/pre-fill/QCfilt_{QC2_descr}/{MODIS_tile}/'
# MCD19_dir_QC3 = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/pre-fill/QCfilt_{QC3_descr}/{MODIS_tile}/'
GCOM_dir_QCbasic = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/GCOM-C/RSRF/CV-MVC/pre-fill/QCfilt_basic/{MODIS_tile}/'

MOD09_basedir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/MODIS/Reflectance_MOD09/CV-MVC/QCfilt_CloudFree_AllAOD_NotSurround'

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
#%%

# MCD19_QC1_arr = []
# MCD19_QC2_arr = []
# MCD19_QC3_arr = []
# MOD13_arr = []
# MCD43_arr = []
# 
# GCOM_arr = []

# # composite_years = np.arange(2000, 2026)

# GCOM_files = sorted(glob.glob(os.path.join(GCOM_dir_QCbasic, '*.nc')))

# # composite_year = 2015
# for composite_year in composite_years:
#     print(f"Loading MCD19 year {composite_year}...")

#     # composite_nn = 12
#     for composite_nn in range(0, 23):
#         print(f"  composite {composite_nn:02d}...")

#         # List all files for this month portion
#         # e.g. MCD19A1.h09v05.061_1km_16day_CV-MVC_QCfilt_CloudFree_AllAOD_AllAdj_2025-21.nc
#         MCD19A1_QC1_files = glob.glob(f'{MCD19_dir_QC1}MCD19A1.{MODIS_tile}.*_{composite_year}-{composite_nn:02d}.nc')
#         MCD19A1_QC2_files = glob.glob(f'{MCD19_dir_QC2}MCD19A1.{MODIS_tile}.*_{composite_year}-{composite_nn:02d}.nc')
#         MCD19A1_QC3_files = glob.glob(f'{MCD19_dir_QC3}MCD19A1.{MODIS_tile}.*_{composite_year}-{composite_nn:02d}.nc')

#         if len(MCD19A1_QC1_files) > 0:
#             MCD19A1_QC1_file = MCD19A1_QC1_files[0]  # Assuming there's only one file per month portion
#             MCD19A1_QC1_ds_curv = xr.open_dataset(MCD19A1_QC1_file)
#             MCD19A1_QC1_ds_pt = clip_2dcurv_ds_to_point(MCD19A1_QC1_ds_curv, site_lon, site_lat)
#             MCD19_QC1_arr.append(MCD19A1_QC1_ds_pt)
#         else:
#             print(f"No files for {composite_year}-{composite_nn:02d} in {MCD19_dir_QC1}")
            
#         if len(MCD19A1_QC2_files) > 0:
#             MCD19A1_QC2_file = MCD19A1_QC2_files[0]  # Assuming there's only one file per month portion
#             MCD19A1_QC2_ds_curv = xr.open_dataset(MCD19A1_QC2_file)
#             MCD19A1_QC2_ds_pt = clip_2dcurv_ds_to_point(MCD19A1_QC2_ds_curv, site_lon, site_lat)
#             MCD19_QC2_arr.append(MCD19A1_QC2_ds_pt)
#         else:
#             print(f"No files for {composite_year}-{composite_nn:02d} in {MCD19_dir_QC2}")

#         if len(MCD19A1_QC3_files) > 0:
#             MCD19A1_QC3_file = MCD19A1_QC3_files[0]  # Assuming there's only one file per month portion
#             MCD19A1_QC3_ds_curv = xr.open_dataset(MCD19A1_QC3_file)
#             MCD19A1_QC3_ds_pt = clip_2dcurv_ds_to_point(MCD19A1_QC3_ds_curv, site_lon, site_lat)
#             MCD19_QC3_arr.append(MCD19A1_QC3_ds_pt)
#         else:
#             print(f"No files for {composite_year}-{composite_nn:02d} in {MCD19_dir_QC3}")

#         # Sample file: GC1SG1_RSRF_h10v04_250m_16day_CV-MVC_QCfilt_basic_2019-20.nc
#         GCOM_file = os.path.join(GCOM_dir_QCbasic, f'GC1SG1_RSRF_{MODIS_tile}_250m_16day_CV-MVC_QCfilt_basic_{composite_year}-{composite_nn:02d}.nc')

#         if not os.path.exists(GCOM_file):
#             print(f"File does not exist: {GCOM_file}")
#             continue
#         GCOM_ds_curv = xr.open_dataset(GCOM_file)
#         GCOM_ds_pt = clip_2dcurv_ds_250m_to_point(GCOM_ds_curv, site_lon, site_lat)

#         # Find the x, y indices where lon_pt and lat_pt are found in ds
#         ds = GCOM_ds_curv
#         lon_2d = ds.lon.values
#         lat_2d = ds.lat.values

#         # Compute the absolute difference and find the indices of the minimum
#         dist = np.abs(lon_2d - site_lon) + np.abs(lat_2d - site_lat)
#         x_idx, y_idx = np.unravel_index(np.argmin(dist), lon_2d.shape)
#         ds_point_2d = ds.isel(x=slice(x_idx-1, x_idx+2), y=slice(y_idx-1, y_idx+2))
#         ds_point = ds_point_2d.mean(dim=['x', 'y'], skipna=True)
#         GCOM_arr.append(GCOM_ds_pt)


# MCD19_QC1_pt = xr.concat(MCD19_QC1_arr, dim='time')
# MCD19_QC2_pt = xr.concat(MCD19_QC2_arr, dim='time')
# MCD19_QC3_pt = xr.concat(MCD19_QC3_arr, dim='time')

# # MOD13_pt = xr.concat(MOD13_arr, dim='time')
# # MCD43_pt = xr.concat(MCD43_arr, dim='time')
# GCOM_pt = xr.concat(GCOM_arr, dim='time')


# # MOD13_pt['time'] = pd.to_datetime([pd.Timestamp(t.isoformat()) if hasattr(t, 'isoformat') else t for t in MOD13_pt.time.values])
# # MCD43_pt['time'] = pd.to_datetime(MCD43_pt.time.values)
# GCOM_pt['time'] = pd.to_datetime(GCOM_pt.time.values)



# ### Compute CCI and NDVI from GCOM

# ## Compute CCI from MCD19
# MCD19_QC1_pt = MCD19_QC1_pt.assign(CCI = (MCD19_QC1_pt.Sur_refl11 - MCD19_QC1_pt.Sur_refl1) / (MCD19_QC1_pt.Sur_refl11 + MCD19_QC1_pt.Sur_refl1))
# MCD19_QC2_pt = MCD19_QC2_pt.assign(CCI = (MCD19_QC2_pt.Sur_refl11 - MCD19_QC2_pt.Sur_refl1) / (MCD19_QC2_pt.Sur_refl11 + MCD19_QC2_pt.Sur_refl1))
# MCD19_QC3_pt = MCD19_QC3_pt.assign(CCI = (MCD19_QC3_pt.Sur_refl11 - MCD19_QC3_pt.Sur_refl1) / (MCD19_QC3_pt.Sur_refl11 + MCD19_QC3_pt.Sur_refl1))
# GCOM_pt = GCOM_pt.assign(NDVI = (GCOM_pt.VN11 - GCOM_pt.VN08) / (GCOM_pt.VN11 + GCOM_pt.VN08))

# ## Compute NDVI from MCD19
# MCD19_QC1_pt = MCD19_QC1_pt.assign(NDVI = (MCD19_QC1_pt.Sur_refl2 - MCD19_QC1_pt.Sur_refl1) / (MCD19_QC1_pt.Sur_refl2 + MCD19_QC1_pt.Sur_refl1))
# MCD19_QC2_pt = MCD19_QC2_pt.assign(NDVI = (MCD19_QC2_pt.Sur_refl2 - MCD19_QC2_pt.Sur_refl1) / (MCD19_QC2_pt.Sur_refl2 + MCD19_QC2_pt.Sur_refl1))
# MCD19_QC3_pt = MCD19_QC3_pt.assign(NDVI = (MCD19_QC3_pt.Sur_refl2 - MCD19_QC3_pt.Sur_refl1) / (MCD19_QC3_pt.Sur_refl2 + MCD19_QC3_pt.Sur_refl1))
# GCOM_pt = GCOM_pt.assign(CCI = (GCOM_pt.VN05 - GCOM_pt.VN08) / (GCOM_pt.VN05 + GCOM_pt.VN08))


#%%

### Load MOD09 data for the same site and time period

MOD09_dir = f'{MOD09_basedir}/{site_name}/'
MOD09_files = sorted(glob.glob(os.path.join(MOD09_dir, '*.nc')))

MOD09_arr = []
for MOD09_file in MOD09_files:

    MOD09_xr = xr.open_dataset(MOD09_file)
    if 'x' in MOD09_xr.dims:
        MOD09_xr = MOD09_xr.squeeze(['x'], drop = True)
    if 'y' in MOD09_xr.dims:
        MOD09_xr = MOD09_xr.squeeze(['y'], drop = True)
    MOD09_arr.append(MOD09_xr)

if len(MOD09_arr) > 0:
    MOD09_pt = xr.concat(MOD09_arr, dim='time', coords='minimal', compat='override').sortby('time')
else:
    MOD09_pt = xr.Dataset()


#%%
# save data to temp
# temp_dat_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/data/temp'
# MCD19_QC1_temp_file = os.path.join(temp_dat_dir, f'MCD19_QCfilt_{QC1_descr}_{site_name}.nc')
# MCD19_QC2_temp_file = os.path.join(temp_dat_dir, f'MCD19_QCfilt_{QC2_descr}_{site_name}.nc')
# MCD19_QC3_temp_file = os.path.join(temp_dat_dir, f'MCD19_QCfilt_{QC3_descr}_{site_name}.nc')
# MCD19_QC1_pt.to_netcdf(MCD19_QC1_temp_file)
# MCD19_QC2_pt.to_netcdf(MCD19_QC2_temp_file)
# MCD19_QC3_pt.to_netcdf(MCD19_QC3_temp_file)
# MOD13_temp_file = os.path.join(temp_dat_dir, f'MOD13_pt_{site_name}.nc')
# MCD43_temp_file = os.path.join(temp_dat_dir, f'MCD43_pt_{site_name}.nc')
# GCOM_temp_file = os.path.join(temp_dat_dir, f'GCOM_pt_{site_name}.nc')
# MOD09_temp_file = os.path.join(temp_dat_dir, f'MOD09_pt_{site_name}.nc')

# MCD19_CloudFree_pt.to_netcdf(MCD19_CloudFree_temp_file)
# MCD19_basicfill_pt.to_netcdf(MCD19_basicfill_temp_file)
# MOD13_pt.to_netcdf(MOD13_temp_file)
# MCD43_pt.to_netcdf(MCD43_temp_file)
# GCOM_pt.to_netcdf(GCOM_temp_file)
# MOD09_pt.to_netcdf(MOD09_temp_file)


#%%
### If already saved, just load it! 


temp_dat_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/data/temp'
MCD19_QC1_temp_file = os.path.join(temp_dat_dir, f'MCD19_QCfilt_{QC1_descr}_{site_name}.nc')
MCD19_QC2_temp_file = os.path.join(temp_dat_dir, f'MCD19_QCfilt_{QC2_descr}_{site_name}.nc')
# MCD19_QC3_temp_file = os.path.join(temp_dat_dir, f'MCD19_QCfilt_{QC3_descr}_{site_name}.nc')
MOD13_temp_file = os.path.join(temp_dat_dir, f'MOD13_pt_{site_name}.nc')
MCD43_temp_file = os.path.join(temp_dat_dir, f'MCD43_pt_{site_name}.nc')
MOD09_temp_file = os.path.join(temp_dat_dir, f'MOD09_pt_{site_name}.nc')
GCOM_temp_file = os.path.join(temp_dat_dir, f'GCOM_pt_{site_name}.nc')
MCD19_QC1_pt = xr.open_dataset(MCD19_QC1_temp_file)
MCD19_QC2_pt = xr.open_dataset(MCD19_QC2_temp_file)
# MCD19_QC3_pt = xr.open_dataset(MCD19_QC3_temp_file)
MOD13_pt = xr.open_dataset(MOD13_temp_file)
MCD43_pt = xr.open_dataset(MCD43_temp_file)
GCOM_pt = xr.open_dataset(GCOM_temp_file)
# MOD09_pt = xr.open_dataset(MOD09_temp_file)



#%%

Photospec_df = pd.read_csv(Photospec_file)
print("Photospec_df columns:", Photospec_df.columns.tolist())

# Print the time range in the 'Time' column of Photospec_df
print(f"Photospec time range: {Photospec_df['Time'].iloc[0]} to {Photospec_df['Time'].iloc[-1]}")

# Remove any data from Photospec_df where NDVI is below 0.6
Photospec_df = Photospec_df[Photospec_df['NDVI'] >= 0.6].reset_index(drop=True)
# Convert Photospec timestamps from "M/D/YY HH:MM" format to datetime64
Photospec_times = pd.to_datetime(Photospec_df['Time'], format='%m/%d/%y %H:%M')


#############################################################
### Load SNODAS 0.05° for approximate snow depth filter
#############################################################

#%%
SNODAS_file = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/SNODAS/masked/05d/SNODAS_WUS_16day_05d_2003-2024.nc'
snow_depth_thresh_mm = 200  #  mm snow depth threshold

snodas_xr_raw = xr.open_dataset(SNODAS_file, decode_coords="all").sortby(['lat', 'lon'], ascending=True)
snodas_xr = snodas_xr_raw['SNWZ'].sel(lon = site_lon, lat = site_lat, method="nearest") # Get snow depth variable
snodas_xr = snodas_xr.sel(time=slice('2017-06-01', '2018-06-30'))
snodas_xr_MODtime = snodas_xr.interp(time=MCD19_QC1_pt.time, method="linear")

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



MCD19_QC1_times = np.array([np.datetime64(t) for t in MCD19_QC1_pt['time'].values])
MCD19_QC2_times = np.array([np.datetime64(t) for t in MCD19_QC2_pt['time'].values])
# MCD19_QC3_times = np.array([np.datetime64(t) for t in MCD19_QC3_pt['time'].values])
MOD13_times = np.array([np.datetime64(t) for t in MOD13_pt['time'].values])
MCD43_times = np.array([np.datetime64(t) for t in MCD43_pt['time'].values])
GCOM_times = np.array([np.datetime64(t) for t in GCOM_pt['time'].values])
MOD09_times = np.array([np.datetime64(t) for t in MOD09_pt['time'].values])  


# Filter Photospec_df for times between 13:00 and 14:00
mask = (Photospec_times.dt.hour >= 13) & (Photospec_times.dt.hour < 14)
Photospec_df_filtered = Photospec_df[mask].reset_index(drop=True)
Photospec_times_filtered = Photospec_times[mask].reset_index(drop=True)

# Set x-axis to month abbreviations at the start of each month
months = np.arange(1, 13)
month_starts = [pd.Timestamp(f'{composite_years[0]}-{m:02d}-01').dayofyear for m in months]
month_labels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']

# Calculate day of year for MCD19_CloudFree_times and Photospec_times_filtered
MCD19_QC1_times = np.array([np.datetime64(t) for t in MCD19_QC1_pt['time'].values])
MCD19_QC2_times = np.array([np.datetime64(t) for t in MCD19_QC2_pt['time'].values])
# MCD19_QC3_times = np.array([np.datetime64(t) for t in MCD19_QC3_pt['time'].values])
MOD13_times = np.array([np.datetime64(t) for t in MOD13_pt['time'].values])
MCD43_times = np.array([np.datetime64(t) for t in MCD43_pt['time'].values])
GCOM_times = np.array([np.datetime64(t) for t in GCOM_pt['time'].values])
MOD09_times = np.array([np.datetime64(t) for t in MOD09_pt['time'].values])
mcd19_doy = pd.to_datetime(MCD19_QC1_times).dayofyear
mod13_doy = pd.to_datetime(MOD13_times).dayofyear
mcd43_doy = pd.to_datetime(MCD43_times).dayofyear
mod09_doy = pd.to_datetime(MOD09_times).dayofyear
photospec_doy = Photospec_times_filtered.dt.dayofyear
GCOM_doy = pd.to_datetime(GCOM_times).dayofyear

# Sort data by day of year so you can plot as lines without weird zigzags
mcd19_doy_sorted = mcd19_doy[np.argsort(mcd19_doy)]
mcd19_ndvi_sorted = MCD19_QC1_pt['NDVI'].values[np.argsort(mcd19_doy)]
GCOM_doy_sorted = GCOM_doy[np.argsort(GCOM_doy)]
GCOM_ndvi_sorted = GCOM_pt['NDVI'].values[np.argsort(GCOM_doy)]
MOD09_doy_sorted = mod09_doy[np.argsort(mod09_doy)]
MOD09_ndvi_sorted = MOD09_pt['NDVI'].values[np.argsort(mod09_doy)]

mod13_doy_sorted = mod13_doy[np.argsort(mod13_doy)]
mod13_ndvi_sorted = MOD13_pt['NDVI'].values[np.argsort(mod13_doy)]
mcd43_doy_sorted = mcd43_doy[np.argsort(mcd43_doy)]
mcd43_ndvi_sorted = MCD43_pt['NDVI'].values[np.argsort(mcd43_doy)]


#%%
# Set global matplotlib font sizes for axes and legends
plt.rcParams.update({
   'axes.labelsize': 15,
   'axes.titlesize': 15,
   'xtick.labelsize': 15,
   'ytick.labelsize': 15,
   'legend.fontsize': 13
})


Photospec_color = 'black'

MCD19_QC1_color = '#BF2665'
MCD19_QC2_color = '#DB830B' # 
GCOM_color = '#56B4E9'
MOD09_color = '#B288E9'
MOD13_color = '#0DA10A'
MCD43_color = '#B5C230'

# Plot shaded regions for winter periods
winter_periods = [
    # Ground snow cover periods
    (pd.Timestamp('2017-09-25'), pd.Timestamp('2018-05-30'))
]

##############################################
## COMPARE NDVI
##############################################


ndvi_arrays = [
    MCD19_QC1_pt['NDVI'].values,
    MCD19_QC2_pt['NDVI'].values,
    MOD13_pt['NDVI'].values,
    MCD43_pt['NDVI'].values,
    GCOM_pt['NDVI'].values,
    MOD09_pt['NDVI'].values
]
mcd19_ndvi_min = np.nanmin(np.concatenate(ndvi_arrays))
mcd19_ndvi_max = np.nanmax(np.concatenate(ndvi_arrays))
mcd19_ndvi_range = mcd19_ndvi_max - mcd19_ndvi_min


xmin = dt(2017, 6, 20)
xmax = dt(2019, 1, 10)
# Plot time series of NDVI from MCD19_CloudFree_pt and MCD19_basicfill_pt
fig, ax = plt.subplots(figsize=(7, 5))

ax.axvspan(snow_exceed_start, snow_exceed_end, color='lightblue', label=f'Snow cover', alpha=0.3)
ax.scatter(Photospec_times_filtered, Photospec_df_filtered['NDVI'], color='black', s=12, alpha=0.8)
ax2 = ax.twinx()

ax2.scatter(MOD13_times, MOD13_pt['NDVI'].values, color=MOD13_color, s=35, alpha=1, label='MOD13 NDVI', marker='s', edgecolors='black', linewidths=0.5)
ax2.scatter(MCD43_times, MCD43_pt['NDVI'].values, color=MCD43_color, s=22, alpha=1, label='MCD43 NDVI', marker='s', edgecolors='black', linewidths=0.5)

# ax2.scatter(GCOM_times, GCOM_pt['NDVI'].values, color=GCOM_color, s=55, alpha=1, label='GCOM NDVI', marker='^', edgecolors='black', linewidths=0.5, zorder = 3)
ax2.scatter(MOD09_times, MOD09_pt['NDVI'].values, color=MOD09_color, s=60, alpha=1, label='MOD09 NDVI', marker='^', edgecolors='black', linewidths=0.5, zorder = 2)


ax2.plot(GCOM_times, GCOM_pt['NDVI'].values, color=GCOM_color, marker = '^', markersize=9, linewidth=2.5, alpha=1, zorder = 2)
# ax2.plot(MOD09_times, MOD09_pt['NDVI'].values, color=MOD09_color, marker = '^', markersize=9, linewidth=2.5, alpha=1, zorder = 2)

ax2.plot(MCD19_QC1_times, MCD19_QC1_pt['NDVI'].values, color=MCD19_QC1_color, marker = 'o', markersize=5, linewidth=2.5, alpha=1, label=QC1_descr, zorder=2)
ax2.plot(MCD19_QC2_times, MCD19_QC2_pt['NDVI'].values, color=MCD19_QC2_color, marker = 'o', markersize=5, linewidth=2.5, alpha=1, label=QC2_descr, zorder=2)


ax2.set_ylim(mcd19_ndvi_min - (mcd19_ndvi_range*0.05), mcd19_ndvi_max + (mcd19_ndvi_range*0.05))

ax.set_ylabel('NDVI')
ax.set_xlim(xmin, xmax)
ax.yaxis.set_major_formatter(FFmt(lambda y, _: f'{y:.1f}'))
ax2.yaxis.set_major_formatter(FFmt(lambda y, _: f'{y:.1f}'))

ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax2.yaxis.set_major_locator(MultipleLocator(0.3))

ax.xaxis.set_major_locator(MonthLocator(bymonth=[7, 10], interval=1))
ax.xaxis.set_major_locator(MonthLocator(bymonth=[1, 4, 7, 10]))
ax.xaxis.set_major_formatter(DateFormatter('%b\n%Y'))
ax.set_title('')



# Combine legends using Patch
handles = [
    Patch(color = 'lightblue', label=f'SNODAS snow depth > {snow_depth_thresh_mm} mm', alpha=0.3),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label='PhotoSpec'),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=MOD13_color, markeredgecolor='black', markersize=7.5, label=f'MOD13'),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=MCD43_color, markeredgecolor='black', markersize=6, label=f'MCD43 (NBAR)'),
    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=MOD09_color, markeredgecolor='black', markersize=10, label=f'MOD09'),
    plt.Line2D([0], [0], color=GCOM_color, marker='^', markersize = 9, linewidth=2.5, label=f'GCOM'),
    plt.Line2D([0], [0], color=MCD19_QC1_color, marker='o', markersize = 5, linewidth=2.5, label=f'MCD19 ({QC1_legend_descr})'),
    plt.Line2D([0], [0], color=MCD19_QC2_color, marker='o', markersize = 5, linewidth=2.5, label=f'MCD19 ({QC2_legend_descr})')
]

ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.2), fontsize=10, ncol=2)
plt.subplots_adjust(bottom=0.35)

# ax.xaxis.set_major_formatter(DateFormatter('%Y'))
# fig.autofmt_xdate(rotation=45)
plt.tight_layout()
plt.show()


#%%

##############################################
## COMPARE CCI
##############################################
cci_arrays = [
    MCD19_QC1_pt['CCI'].values,
    MCD19_QC2_pt['CCI'].values,
    GCOM_pt['CCI'].values,
    # MOD09_pt['CCI'].values
]
mcd19_cci_min = np.nanmin(np.concatenate(cci_arrays))
mcd19_cci_max = np.nanmax(np.concatenate(cci_arrays))
mcd19_cci_range = mcd19_cci_max - mcd19_cci_min


xmin = dt(2017, 6, 20)
xmax = dt(2019, 1, 10)
# Plot time series of CCI from MCD19_CloudFree_pt and MCD19_basicfill_pt
fig, ax = plt.subplots(figsize=(7, 5))

ax.axvspan(snow_exceed_start, snow_exceed_end, color='lightblue', label=f'Snow cover', alpha=0.3)
ax.scatter(Photospec_times_filtered, Photospec_df_filtered['CCI'], color='black', s=12, alpha=0.8)
ax2 = ax.twinx()

ax2.scatter(MOD09_times, MOD09_pt['CCI'].values, color=MOD09_color, s=60, alpha=1, label='MOD09 CCI', marker='^', edgecolors='black', linewidths=0.5, zorder = 2)
ax2.plot(GCOM_times, GCOM_pt['CCI'].values, color=GCOM_color, marker = '^', markersize=9, linewidth=2.5, alpha=1, zorder = 2)
ax2.plot(MCD19_QC1_times, MCD19_QC1_pt['CCI'].values, color=MCD19_QC1_color, marker = 'o', markersize=5, linewidth=2.5, alpha=1, label=QC1_descr, zorder=2)
ax2.plot(MCD19_QC2_times, MCD19_QC2_pt['CCI'].values, color=MCD19_QC2_color, marker = 'o', markersize=5, linewidth=2.5, alpha=1, label=QC2_descr, zorder=2)


ax2.set_ylim(mcd19_cci_min - (mcd19_cci_range*0.05), mcd19_cci_max + (mcd19_cci_range*0.05))

ax.set_ylabel('CCI')
ax.set_xlim(xmin, xmax)
ax.set_ylim(-0.05, 0.4)
ax.yaxis.set_major_formatter(FFmt(lambda y, _: f'{y:.1f}'))
ax2.yaxis.set_major_formatter(FFmt(lambda y, _: f'{y:.1f}'))
ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax2.yaxis.set_major_locator(MultipleLocator(0.3))
ax.xaxis.set_major_locator(MonthLocator(bymonth=[7, 10], interval=1))
ax.xaxis.set_major_locator(MonthLocator(bymonth=[1, 4, 7, 10]))
ax.xaxis.set_major_formatter(DateFormatter('%b\n%Y'))
ax.set_title('')



# Combine legends using Patch
handles = [
    Patch(color = 'lightblue', label=f'SNODAS snow depth > {snow_depth_thresh_mm} mm', alpha=0.3),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label='PhotoSpec'),
    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=MOD09_color, markeredgecolor='black', markersize=10, label=f'MOD09'),
    plt.Line2D([0], [0], color=GCOM_color, marker='^', markersize = 9, linewidth=2.5, label=f'GCOM'),
    plt.Line2D([0], [0], color=MCD19_QC1_color, marker='o', markersize = 5, linewidth=2.5, label=f'MCD19 ({QC1_legend_descr})'),
    plt.Line2D([0], [0], color=MCD19_QC2_color, marker='o', markersize = 5, linewidth=2.5, label=f'MCD19 ({QC2_legend_descr})')
]
ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.2), fontsize=10, ncol=2)
plt.subplots_adjust(bottom=0.35)

# ax.xaxis.set_major_formatter(DateFormatter('%Y'))
# fig.autofmt_xdate(rotation=45)
plt.tight_layout()
plt.show()



#%%


##############################################
## COMPARE Band 1 (Red) Reflectance
##############################################

refl_arrays = [
    MCD19_QC1_pt['Sur_refl1'].values,
    MCD19_QC2_pt['Sur_refl1'].values,
    MCD43_pt['B1'].values,
    GCOM_pt['VN08'].values,
    MOD09_pt['sur_refl_b01'].values
]
mcd19_refl_min = np.nanmin(np.concatenate(refl_arrays))
mcd19_refl_max = np.nanmax(np.concatenate(refl_arrays))
mcd19_refl_range = mcd19_refl_max - mcd19_refl_min


# Plot time series of NDVI from MCD19_CloudFree_pt and MCD19_basicfill_pt
fig, ax = plt.subplots(figsize=(7, 5))

ax.axvspan(snow_exceed_start, snow_exceed_end, color='lightblue', label=f'Snow cover', alpha=0.3)

ax.scatter(MCD43_times, MCD43_pt['B1'].values, color=MCD43_color, s=40, alpha=0.7,  marker='s', edgecolors='black', linewidths=0.5)
ax.scatter(MOD09_times, MOD09_pt['sur_refl_b01'].values, color=MOD09_color, s=55, alpha=1,  marker='^', edgecolors='black', linewidths=0.5)
ax.plot(GCOM_times, GCOM_pt['VN08'].values, color=GCOM_color, marker = '^', markersize=9, linewidth=2.5, alpha=1, zorder = 2)
ax.plot(MCD19_QC1_times, MCD19_QC1_pt['Sur_refl1'].values, color=MCD19_QC1_color, marker = 'o', markersize=5, linewidth=2.5, alpha=1, zorder=2)
ax.plot(MCD19_QC2_times, MCD19_QC2_pt['Sur_refl1'].values, color=MCD19_QC2_color, marker = 'o', markersize=5, linewidth=2.5, alpha=1, zorder=2)


ax.set_ylim(mcd19_refl_min - (mcd19_refl_range*0.05), mcd19_refl_max + (mcd19_refl_range*0.05))

ax.set_ylabel('Red Reflectance')
ax.set_xlim(xmin, xmax)
ax.yaxis.set_major_formatter(FFmt(lambda y, _: f'{y:.1f}'))
ax.yaxis.set_major_formatter(FFmt(lambda y, _: f'{y:.1f}'))
ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax.xaxis.set_major_locator(MonthLocator(bymonth=[7, 10], interval=1))
ax.xaxis.set_major_locator(MonthLocator(bymonth=[1, 4, 7, 10]))
ax.xaxis.set_major_formatter(DateFormatter('%b\n%Y'))
ax.set_title('')



# Combine legends using Patch
handles = [
    Patch(color = 'lightblue', label=f'SNODAS snow depth > {snow_depth_thresh_mm} mm', alpha=0.3),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=MCD43_color, markeredgecolor='black', markersize=6, label=f'MCD43 (NBAR)'),
    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=MOD09_color, markeredgecolor='black', markersize=10, label=f'MOD09'),
    plt.Line2D([0], [0], color=GCOM_color, marker='^', markersize = 9, linewidth=2.5, label=f'GCOM'),
    plt.Line2D([0], [0], color=MCD19_QC1_color, marker='o', markersize = 5, linewidth=2.5, label=f'MCD19 ({QC1_legend_descr})'),
    plt.Line2D([0], [0], color=MCD19_QC2_color, marker='o', markersize = 5, linewidth=2.5, label=f'MCD19 ({QC2_legend_descr})')
]
ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.2), fontsize=10, ncol=2)
plt.subplots_adjust(bottom=0.35)

# ax.xaxis.set_major_formatter(DateFormatter('%Y'))
# fig.autofmt_xdate(rotation=45)
plt.tight_layout()
plt.show()


#%%



##############################################
## COMPARE Band 2 (NIR) Reflectance
##############################################

refl_arrays = [
    MCD19_QC1_pt['Sur_refl2'].values,
    MCD19_QC2_pt['Sur_refl2'].values,
    MCD43_pt['B2'].values,
    GCOM_pt['VN05'].values,
    MOD09_pt['sur_refl_b02'].values
]
mcd19_refl_min = np.nanmin(np.concatenate(refl_arrays))
mcd19_refl_max = np.nanmax(np.concatenate(refl_arrays))
mcd19_refl_range = mcd19_refl_max - mcd19_refl_min


# Plot time series of NDVI from MCD19_CloudFree_pt and MCD19_basicfill_pt
fig, ax = plt.subplots(figsize=(7, 5))

ax.axvspan(snow_exceed_start, snow_exceed_end, color='lightblue', label=f'Snow cover', alpha=0.3)

ax.scatter(MCD43_times, MCD43_pt['B2'].values, color=MCD43_color, s=40, alpha=0.7,  marker='s', edgecolors='black', linewidths=0.5)
ax.scatter(MOD09_times, MOD09_pt['sur_refl_b02'].values, color=MOD09_color, s=55, alpha=1, marker='^', edgecolors='black', linewidths=0.5)

ax.plot(GCOM_times, GCOM_pt['VN05'].values, color=GCOM_color, marker = '^', markersize=9, linewidth=2.5, alpha=1, zorder = 2)
ax.plot(MCD19_QC1_times, MCD19_QC1_pt['Sur_refl2'].values, color=MCD19_QC1_color, marker = 'o', markersize=5, linewidth=2.5, alpha=1, zorder=2)
ax.plot(MCD19_QC2_times, MCD19_QC2_pt['Sur_refl2'].values, color=MCD19_QC2_color, marker = 'o', markersize=5, linewidth=2.5, alpha=1, zorder=2)


ax.set_ylim(mcd19_refl_min - (mcd19_refl_range*0.05), mcd19_refl_max + (mcd19_refl_range*0.05))

ax.set_ylabel('NIR Reflectance')
ax.set_xlim(xmin, xmax)
ax.yaxis.set_major_formatter(FFmt(lambda y, _: f'{y:.1f}'))
ax.yaxis.set_major_locator(MultipleLocator(0.1))

ax.xaxis.set_major_locator(MonthLocator(bymonth=[7, 10], interval=1))
ax.xaxis.set_major_locator(MonthLocator(bymonth=[1, 4, 7, 10]))
ax.xaxis.set_major_formatter(DateFormatter('%b\n%Y'))
ax.set_title('')



# Combine legends using Patch
handles = [
    Patch(color = 'lightblue', label=f'SNODAS snow depth > {snow_depth_thresh_mm} mm', alpha=0.3),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=MCD43_color, markeredgecolor='black', markersize=6, label=f'MCD43 (NBAR)'),
    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=MOD09_color, markeredgecolor='black', markersize=10, label=f'MOD09'),
    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=GCOM_color, markeredgecolor='black', markersize=10, label=f'GCOM'),
    plt.Line2D([0], [0], color=MCD19_QC1_color, marker='o', markersize = 5, linewidth=2.5, label=f'MCD19 ({QC1_legend_descr})'),
    plt.Line2D([0], [0], color=MCD19_QC2_color, marker='o', markersize = 5, linewidth=2.5, label=f'MCD19 ({QC2_legend_descr})')
]
ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.2), fontsize=10, ncol=2)
plt.subplots_adjust(bottom=0.35)

# ax.xaxis.set_major_formatter(DateFormatter('%Y'))
# fig.autofmt_xdate(rotation=45)
plt.tight_layout()
plt.show()



#%%



##############################################
## COMPARE Band 11 Reflectance
##############################################

refl_arrays = [
    MCD19_QC1_pt['Sur_refl11'].values,
    MCD19_QC2_pt['Sur_refl11'].values,
    GCOM_pt['VN11'].values,
    MOD09_pt['sur_refl_b11'].values
]
mcd19_refl_min = np.nanmin(np.concatenate(refl_arrays))
mcd19_refl_max = np.nanmax(np.concatenate(refl_arrays))
mcd19_refl_range = mcd19_refl_max - mcd19_refl_min


# Plot time series of NDVI from MCD19_CloudFree_pt and MCD19_basicfill_pt
fig, ax = plt.subplots(figsize=(7, 5))

ax.axvspan(snow_exceed_start, snow_exceed_end, color='lightblue', label=f'Snow cover', alpha=0.3)

ax.scatter(MOD09_times, MOD09_pt['sur_refl_b11'].values, color=MOD09_color, s=55, alpha=1, marker='^', edgecolors='black', linewidths=0.5)

ax.plot(GCOM_times, GCOM_pt['VN11'].values, color=GCOM_color, marker = '^', markersize=9, linewidth=2.5, alpha=1, zorder = 2)
ax.plot(MCD19_QC1_times, MCD19_QC1_pt['Sur_refl11'].values, color=MCD19_QC1_color, marker = 'o', markersize=5, linewidth=2.5, alpha=1, zorder=2)
ax.plot(MCD19_QC2_times, MCD19_QC2_pt['Sur_refl11'].values, color=MCD19_QC2_color, marker = 'o', markersize=5, linewidth=2.5, alpha=1, zorder=2)


ax.set_ylim(mcd19_refl_min - (mcd19_refl_range*0.05), mcd19_refl_max + (mcd19_refl_range*0.05))

ax.set_ylabel('Green (Band 11) Reflectance')
ax.set_xlim(xmin, xmax)
ax.yaxis.set_major_formatter(FFmt(lambda y, _: f'{y:.1f}'))
ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax.xaxis.set_major_locator(MonthLocator(bymonth=[7, 10], interval=1))
ax.xaxis.set_major_locator(MonthLocator(bymonth=[1, 4, 7, 10]))
ax.xaxis.set_major_formatter(DateFormatter('%b\n%Y'))
ax.set_title('')



# Combine legends using Patch
handles = [
    Patch(color = 'lightblue', label=f'SNODAS snow depth > {snow_depth_thresh_mm} mm', alpha=0.3),
    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=MOD09_color, markeredgecolor='black', markersize=10, label=f'MOD09'),
    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=GCOM_color, markeredgecolor='black', markersize=10, label=f'GCOM'),
    plt.Line2D([0], [0], color=MCD19_QC1_color, marker='o', markersize = 5, linewidth=2.5, label=f'MCD19 ({QC1_legend_descr})'),
    plt.Line2D([0], [0], color=MCD19_QC2_color, marker='o', markersize = 5, linewidth=2.5, label=f'MCD19 ({QC2_legend_descr})')
]
ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.2), fontsize=10, ncol=2)
plt.subplots_adjust(bottom=0.35)

# ax.xaxis.set_major_formatter(DateFormatter('%Y'))
# fig.autofmt_xdate(rotation=45)
plt.tight_layout()
plt.show()


