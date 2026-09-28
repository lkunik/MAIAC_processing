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
site_name = 'OSBS'
MODIS_tile = 'h10v06'



# Compare AOD and adjacency masks:
# 1: CloudFree_AllAOD_AllAdj
# 2: CloudFree_LowAOD_AllAdj

# 3: CloudFree_AllAOD_NotSurround
# 4: CloudFree_LowAOD_NotSurround

# 5: CloudFree_AllAOD_ClearAdj
# 6: CloudFree_LowAOD_ClearAdj

QC1_descr = "CloudFree_AllAOD_AllAdj"
QC2_descr = "CloudFree_AllAOD_NotSurround"
QC3_descr = "CloudFree_AllAOD_ClearAdj"

MCD19_dir_QC1 = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/pre-fill/QCfilt_{QC1_descr}/{MODIS_tile}/'
MCD19_dir_QC2 = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/pre-fill/QCfilt_{QC2_descr}/{MODIS_tile}/'
MCD19_dir_QC3 = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/pre-fill/QCfilt_{QC3_descr}/{MODIS_tile}/'

Photospec_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/data/PhotoSpec'
Photospec_file = os.path.join(Photospec_dir, f'PhotoSpec_{site_name}.csv')


MOD13_file = f'/uufs/chpc.utah.edu/common/home/lin-group23/ltk/MODIS/NDVI_EVI_MOD13A2/from_GEE/{site_name}/modis_ndvi_timeseries_{site_name}.csv'
MCD43_file = f'/uufs/chpc.utah.edu/common/home/lin-group23/ltk/MODIS/Reflectance_MCD43A4/from_GEE/{site_name}/mcd43a4_reflectance_{site_name}.csv'


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
    # print(f"Nearest point in dataset to ({lon_pt}, {lat_pt}) is at ({lon_selected}, {lat_selected}) with distance {dist[y_idx, x_idx]:.6f}")
    ds_point = ds.isel(x=x_idx, y=y_idx)
    return ds_point

### Load all the MCD19 data
#%%

MCD19_QC1_arr = []
MCD19_QC2_arr = []
MCD19_QC3_arr = []
MOD13_arr = []
MCD43_arr = []
composite_year = 2015
for composite_year in composite_years:
    print(f"Loading MCD19 year {composite_year}...")

    composite_nn = 12
    for composite_nn in range(0, 23):
        print(f"  composite {composite_nn:02d}...")

        # List all files for this month portion
        # e.g. MCD19A1.h09v05.061_1km_16day_CV-MVC_QCfilt_CloudFree_AllAOD_AllAdj_2025-21.nc
        MCD19A1_QC1_files = glob.glob(f'{MCD19_dir_QC1}MCD19A1.{MODIS_tile}.*_{composite_year}-{composite_nn:02d}.nc')
        MCD19A1_QC2_files = glob.glob(f'{MCD19_dir_QC2}MCD19A1.{MODIS_tile}.*_{composite_year}-{composite_nn:02d}.nc')
        MCD19A1_QC3_files = glob.glob(f'{MCD19_dir_QC3}MCD19A1.{MODIS_tile}.*_{composite_year}-{composite_nn:02d}.nc')

        if len(MCD19A1_QC1_files) > 0:
            MCD19A1_QC1_file = MCD19A1_QC1_files[0]  # Assuming there's only one file per month portion
            MCD19A1_QC1_ds_curv = xr.open_dataset(MCD19A1_QC1_file)
            MCD19A1_QC1_ds_pt = clip_2dcurv_ds_to_point(MCD19A1_QC1_ds_curv, site_lon, site_lat)
            MCD19_QC1_arr.append(MCD19A1_QC1_ds_pt)
        else:
            print(f"No files for {composite_year}-{composite_nn:02d} in {MCD19_dir_QC1}")
            
        if len(MCD19A1_QC2_files) > 0:
            MCD19A1_QC2_file = MCD19A1_QC2_files[0]  # Assuming there's only one file per month portion
            MCD19A1_QC2_ds_curv = xr.open_dataset(MCD19A1_QC2_file)
            MCD19A1_QC2_ds_pt = clip_2dcurv_ds_to_point(MCD19A1_QC2_ds_curv, site_lon, site_lat)
            MCD19_QC2_arr.append(MCD19A1_QC2_ds_pt)
        else:
            print(f"No files for {composite_year}-{composite_nn:02d} in {MCD19_dir_QC2}")

        if len(MCD19A1_QC3_files) > 0:
            MCD19A1_QC3_file = MCD19A1_QC3_files[0]  # Assuming there's only one file per month portion
            MCD19A1_QC3_ds_curv = xr.open_dataset(MCD19A1_QC3_file)
            MCD19A1_QC3_ds_pt = clip_2dcurv_ds_to_point(MCD19A1_QC3_ds_curv, site_lon, site_lat)
            MCD19_QC3_arr.append(MCD19A1_QC3_ds_pt)
        else:
            print(f"No files for {composite_year}-{composite_nn:02d} in {MCD19_dir_QC3}")


MCD19_QC1_pt = xr.concat(MCD19_QC1_arr, dim='time')
MCD19_QC2_pt = xr.concat(MCD19_QC2_arr, dim='time')
MCD19_QC3_pt = xr.concat(MCD19_QC3_arr, dim='time')


## Compute CCI from MCD19
MCD19_QC1_pt = MCD19_QC1_pt.assign(CCI = (MCD19_QC1_pt.Sur_refl11 - MCD19_QC1_pt.Sur_refl1) / (MCD19_QC1_pt.Sur_refl11 + MCD19_QC1_pt.Sur_refl1))
MCD19_QC2_pt = MCD19_QC2_pt.assign(CCI = (MCD19_QC2_pt.Sur_refl11 - MCD19_QC2_pt.Sur_refl1) / (MCD19_QC2_pt.Sur_refl11 + MCD19_QC2_pt.Sur_refl1))
MCD19_QC3_pt = MCD19_QC3_pt.assign(CCI = (MCD19_QC3_pt.Sur_refl11 - MCD19_QC3_pt.Sur_refl1) / (MCD19_QC3_pt.Sur_refl11 + MCD19_QC3_pt.Sur_refl1))

## Compute NDVI from MCD19
MCD19_QC1_pt = MCD19_QC1_pt.assign(NDVI = (MCD19_QC1_pt.Sur_refl2 - MCD19_QC1_pt.Sur_refl1) / (MCD19_QC1_pt.Sur_refl2 + MCD19_QC1_pt.Sur_refl1))
MCD19_QC2_pt = MCD19_QC2_pt.assign(NDVI = (MCD19_QC2_pt.Sur_refl2 - MCD19_QC2_pt.Sur_refl1) / (MCD19_QC2_pt.Sur_refl2 + MCD19_QC2_pt.Sur_refl1))
MCD19_QC3_pt = MCD19_QC3_pt.assign(NDVI = (MCD19_QC3_pt.Sur_refl2 - MCD19_QC3_pt.Sur_refl1) / (MCD19_QC3_pt.Sur_refl2 + MCD19_QC3_pt.Sur_refl1))


# save data to temp
temp_dat_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/data/temp'
MCD19_QC1_temp_file = os.path.join(temp_dat_dir, f'MCD19_QCfilt_{QC1_descr}_{site_name}.nc')
MCD19_QC2_temp_file = os.path.join(temp_dat_dir, f'MCD19_QCfilt_{QC2_descr}_{site_name}.nc')
MCD19_QC3_temp_file = os.path.join(temp_dat_dir, f'MCD19_QCfilt_{QC3_descr}_{site_name}.nc')
MCD19_QC1_pt.to_netcdf(MCD19_QC1_temp_file)
MCD19_QC2_pt.to_netcdf(MCD19_QC2_temp_file)
MCD19_QC3_pt.to_netcdf(MCD19_QC3_temp_file)
# If already loaded, just read from temp
temp_dat_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/data/temp'
# MCD19_QC1_temp_file = os.path.join(temp_dat_dir, f'MCD19_QCfilt_{QC1_descr}_{site_name}.nc')
# MCD19_QC2_temp_file = os.path.join(temp_dat_dir, f'MCD19_QCfilt_{QC2_descr}_{site_name}.nc')
# MCD19_QC3_temp_file = os.path.join(temp_dat_dir, f'MCD19_QCfilt_{QC3_descr}_{site_name}.nc')
MOD13_temp_file = os.path.join(temp_dat_dir, f'MOD13_pt_{site_name}.nc')
MCD43_temp_file = os.path.join(temp_dat_dir, f'MCD43_pt_{site_name}.nc')
# MCD19_QC1_pt = xr.open_dataset(MCD19_QC1_temp_file)
# MCD19_QC2_pt = xr.open_dataset(MCD19_QC2_temp_file)
# MCD19_QC3_pt = xr.open_dataset(MCD19_QC3_temp_file)

#%%

Photospec_df = pd.read_csv(Photospec_file)
print("Photospec_df columns:", Photospec_df.columns.tolist())

# Print the time range in the 'Time' column of Photospec_df
print(f"Photospec time range: {Photospec_df['Time'].iloc[0]} to {Photospec_df['Time'].iloc[-1]}")

# Remove any data from Photospec_df where NDVI is below 0.5
Photospec_df = Photospec_df[Photospec_df['NDVI'] >= 0.5].reset_index(drop=True)
# Convert Photospec timestamps from "M/D/YY HH:MM" format to datetime64
Photospec_times = pd.to_datetime(Photospec_df['Time'], format='%m/%d/%y %H:%M')

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

# Select MCD19 data within the time range of Photospec data
MCD19_QC1_pt = MCD19_QC1_pt.sel(time=slice(Photospec_times.iloc[0], Photospec_times.iloc[-1]))
MCD19_QC2_pt = MCD19_QC2_pt.sel(time=slice(Photospec_times.iloc[0], Photospec_times.iloc[-1]))
MCD19_QC3_pt = MCD19_QC3_pt.sel(time=slice(Photospec_times.iloc[0], Photospec_times.iloc[-1]))

#############################################################
### Load SNODAS 0.05° for approximate snow depth filter
#############################################################

#%%
MCD19_QC1_times = np.array([np.datetime64(t) for t in MCD19_QC1_pt['time'].values])
MCD19_QC2_times = np.array([np.datetime64(t) for t in MCD19_QC2_pt['time'].values])
MCD19_QC3_times = np.array([np.datetime64(t) for t in MCD19_QC3_pt['time'].values])


# Filter Photospec_df for times between 13:00 and 14:00
mask = (Photospec_times.dt.hour >= 13) & (Photospec_times.dt.hour < 14)
Photospec_df_filtered = Photospec_df[mask].reset_index(drop=True)
Photospec_times_filtered = Photospec_times[mask].reset_index(drop=True)

# Set x-axis to month abbreviations at the start of each month
months = np.arange(1, 13)
month_starts = [pd.Timestamp(f'{composite_years[0]}-{m:02d}-01').dayofyear for m in months]
month_labels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']


# Calculate day of year for MCD19_CloudFree_times and Photospec_times_filtered
mcd19_doy = pd.to_datetime(MCD19_QC1_times).dayofyear
mod13_doy = MOD13_times.dt.dayofyear
mcd43_doy = MCD43_times_8day.dt.dayofyear
photospec_doy = Photospec_times_filtered.dt.dayofyear

# Sort data by day of year so you can plot as lines without weird zigzags
mcd19_doy_sorted = mcd19_doy[np.argsort(mcd19_doy)]
mcd19_ndvi_sorted = MCD19_QC1_pt['NDVI'].values[np.argsort(mcd19_doy)]

mod13_doy_sorted = mod13_doy[np.argsort(mod13_doy)]
mod13_ndvi_sorted = MOD13_df['ndvi'].values[np.argsort(mod13_doy)]
mcd43_doy_sorted = mcd43_doy[np.argsort(mcd43_doy)]
mcd43_ndvi_sorted = MCD43_df_8day['ndvi'].values[np.argsort(mcd43_doy)]


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
MCD43_color = '#E69F00'

fig, ax = plt.subplots(figsize=(7, 3))

# # Plot shaded region where snow depth exceeds threshold
# ax.axvspan(first_highsnow_doy - 15, last_highsnow_doy + 15, color='lightblue', label=f'SNODAS snow depth > {snow_depth_thresh_mm} mm', alpha=0.5)

# Plot Photospec NDVI on left y-axis
ax.scatter(photospec_doy, Photospec_df_filtered['NDVI'], label='Photospec', color='black', s=10, alpha=0.7)
ax.scatter(mod13_doy_sorted, mod13_ndvi_sorted, label='MOD13 NDVI', color=MOD13_color, s=20, alpha=1)
ax.scatter(mcd19_doy_sorted, mcd19_ndvi_sorted, label=f'MCD19 NDVI\n({QC1_descr})', color=MCD19_color, s=20, alpha=1)
ax.scatter(mcd43_doy_sorted, mcd43_ndvi_sorted, label='MCD43 NDVI', color=MCD43_color, s=20, alpha=1)


# Adjust limits and legend
ax.set_ylabel('NDVI')
ax.set_xticks(month_starts)
ax.set_xticklabels(month_labels)


# Combine legends using Patch
handles = [
    # Patch(color='lightblue', label=f'SNODAS snow\ndepth > {snow_depth_thresh_mm} mm', alpha=0.5),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label='Photospec'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=MCD19_color, markersize=8, label='MCD19'),
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
# ax.axvspan(snow_exceed_start, snow_exceed_end, color='lightblue', label=f'Snow cover', alpha=0.3)

# Plot Photospec NDVI on left y-axis
ax.scatter(Photospec_times_filtered, Photospec_df_filtered['NDVI'], label='Photospec', color='black', s=10, alpha=0.7)
ax.scatter(MOD13_times, MOD13_df['ndvi'], label='MOD13 NDVI', color=MOD13_color, s=28, alpha=1)
ax.scatter(MCD19_QC1_times, MCD19_QC1_pt['NDVI'].values, label=f'MCD19 NDVI\n({QC1_descr})', color=MCD19_color, s=28, alpha=1)
ax.scatter(MCD43_times_8day, MCD43_df_8day['ndvi'], label='MCD43 NDVI', color=MCD43_color, s=28, alpha=1)

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
fig, axes = plt.subplots(2, 1, figsize=(7, 4.7), sharex=True)

# --- Top subplot: NDVI comparison ---
ax = axes[0]
# ax.axvspan(first_highsnow_doy - 15, last_highsnow_doy + 15, color='lightblue', label=f'SNODAS snow depth > {snow_depth_thresh_mm} mm', alpha=0.5)
ax.scatter(photospec_doy, Photospec_df_filtered['NDVI'], label='Photospec', color='black', s=10, alpha=0.7)
ax.scatter(mod13_doy_sorted, mod13_ndvi_sorted, label='MOD13 NDVI', color=MOD13_color, s=28, alpha=1)
ax.scatter(mcd19_doy_sorted, mcd19_ndvi_sorted, label='MCD19 NDVI', color=MCD19_color, s=28, alpha=1)
ax.scatter(mcd43_doy_sorted, mcd43_ndvi_sorted, label='MCD43 NDVI', color=MCD43_color, s=28, alpha=1)
ax.set_ylabel('NDVI')
ax.set_xticks(month_starts)
ax.set_xticklabels(month_labels)
ax.set_title(site_name)
handles = [
    # Patch(color='lightblue', label=f'SNODAS snow\ndepth > {snow_depth_thresh_mm} mm', alpha=0.5),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label='Photospec'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=MCD19_color, markersize=8, label='MCD19'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=MOD13_color, markersize=8, label='MOD13'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=MCD43_color, markersize=8, label='MCD43')
]
ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.05, 1))

# --- Bottom subplot: CCI comparison ---
ax2 = axes[1]
# ax2.axvspan(first_highsnow_doy - 15, last_highsnow_doy + 15, color='lightblue', alpha=0.5)
ax2.scatter(photospec_doy, Photospec_df_filtered['CCI'], label='Photospec CCI', color='black', s=10, alpha=0.7)
ax2.scatter(mcd19_doy_sorted, MCD19_QC1_pt['CCI'].values[np.argsort(mcd19_doy)], label=f'MCD19 CCI\n({QC1_descr})', color=MCD19_color, s=28, alpha=1)
ax2.set_ylabel('CCI')
ax2.set_xticks(month_starts)
ax2.set_xticklabels(month_labels)
handles2 = [
    # Patch(color='lightblue', label=f'SNODAS snow\ndepth > {snow_depth_thresh_mm} mm', alpha=0.5),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label='Photospec'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=MCD19_color, markersize=8, label='MCD19')
]
ax2.legend(handles=handles2, loc='upper left', bbox_to_anchor=(1.05, 1))

plt.tight_layout()
plt.show()

# %%

# Calculate day of year for MCD19_CloudFree_times and Photospec_times_filtered
mcd19_doy = pd.to_datetime(MCD19_QC1_times).dayofyear
photospec_doy = Photospec_times_filtered.dt.dayofyear

plt.figure(figsize=(9, 2.5))
ax = plt.gca()
ax2 = ax.twinx()


# Plot Photospec CCI on left y-axis
ax.scatter(photospec_doy, Photospec_df_filtered['CCI'], label='Photospec', color='black', s=10, alpha=0.7)
ax.set_ylabel('Photospec CCI')

# Plot MCD19 CCI on right y-axis
ax2.scatter(mcd19_doy, MCD19_QC1_pt['CCI'].values, label=f'MCD19 CCI\n({QC1_descr})', color=MCD19_color, s=28, alpha=1)
ax2.set_ylabel('MCD19 CCI')


# Adjust limits and legend
ax.set_xticks(month_starts)
ax.set_xticklabels(month_labels)


# Combine legends using Patch
handles = [
    # Patch(color = 'lightblue', label=f'SNODAS snow\ndepth > {snow_depth_thresh_mm} mm', alpha=0.3),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label='PhotoSpec'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label=f'MCD19\n{QC1_descr}'),
]
ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.23, 1))

plt.tight_layout()
plt.show()


#%%



# Calculate day of year for MCD19_CloudFree_times and Photospec_times_filtered
mcd19_doy = pd.to_datetime(MCD19_QC1_times).dayofyear
photospec_doy = Photospec_times_filtered.dt.dayofyear

fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True, sharey=True)
panel_labels = [ 
    QC1_descr,
    QC2_descr,
    QC3_descr
]
cci_arrays = [
    MCD19_QC1_pt['CCI'].values,
    MCD19_QC2_pt['CCI'].values,
    MCD19_QC3_pt['CCI'].values
]
mcd19_cci_min = np.nanmin(cci_arrays)
mcd19_cci_max = np.nanmax(cci_arrays)
mcd19_cci_range = mcd19_cci_max - mcd19_cci_min
colors = ['red', '#08A0FF', 'orange']

for i, ax in enumerate(axes):
    # ax.axvspan(first_highsnow_doy - 15, last_highsnow_doy + 15, color='lightblue', alpha=0.3)
    ax.scatter(photospec_doy, Photospec_df_filtered['CCI'], color='black', s=10, alpha=0.7, label='Photospec')
    ax2 = ax.twinx()
    ax2.scatter(mcd19_doy, cci_arrays[i], color=colors[i], s=35, alpha=1, label=panel_labels[i])
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
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label='PhotoSpec'),
        ]
        for j in range(len(colors)):
            handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[j], markersize=8, label=panel_labels[j]))
        ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.25, 1))

plt.tight_layout()
plt.show()


#%%



# Calculate day of year for MCD19_CloudFree_times and Photospec_times_filtered

fig, axes = plt.subplots(3, 1, figsize=(5.6, 5), sharex=True, sharey=True)
panel_labels = [ 
    QC1_descr,
    QC2_descr,
    QC3_descr
]
cci_arrays = [
    MCD19_QC1_pt['CCI'].values,
    MCD19_QC2_pt['CCI'].values,
    MCD19_QC3_pt['CCI'].values
]
mcd19_cci_min = np.nanmin(cci_arrays)
mcd19_cci_max = np.nanmax(cci_arrays)
mcd19_cci_range = mcd19_cci_max - mcd19_cci_min
colors = ['red', '#08A0FF', 'orange']

for i, ax in enumerate(axes):
    ax.scatter(Photospec_times_filtered, Photospec_df_filtered['CCI'], color='black', s=10, alpha=0.7, label='Photospec')
    ax2 = ax.twinx()
    ax2.scatter(MCD19_QC1_times, cci_arrays[i], color=colors[i], s=35, alpha=1, label=panel_labels[i])
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



# Apply snow filter: set CCI to nan where snow depth exceeds threshold
MCD19_QC1_snowfilt = MCD19_QC1_pt.copy()

MCD19_QC2_snowfilt = MCD19_QC2_pt.copy()


MCD19_QC3_snowfilt = MCD19_QC3_pt.copy()





#%%

################################################################
### Generate linear correlations between Photospec CCI and snow-filtered MCD19 CCI
################################################################

# First, Interpolate Photospec CCI to match MCD19_QC1_snowfilt.time
photospec_cci_interp = pd.Series(
    np.interp(
        pd.to_datetime(MCD19_QC1_snowfilt.time.values).astype(np.int64),
        Photospec_times_filtered.astype(np.int64),
        Photospec_df_filtered['CCI'].values
    ),
    index=MCD19_QC1_snowfilt.time.values
)


# Set to nan before first and after last valid Photospec time
first_time = Photospec_times_filtered.iloc[0]
last_time = Photospec_times_filtered.iloc[-1]
interp_times = pd.to_datetime(MCD19_QC1_snowfilt.time.values)

photospec_cci_interp[(interp_times < first_time) | (interp_times > last_time)] = np.nan

# Convert interp_times to dayofyear
interp_times_doy = pd.to_datetime(interp_times).dayofyear




# %%


# Calculate day of year for MCD19_CloudFree_times and Photospec_times_filtered
mcd19_doy = pd.to_datetime(MCD19_QC1_times).dayofyear
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
valid_mask = (~np.isnan(MCD19_QC1_snowfilt['CCI'].values)) & (~np.isnan(photospec_cci_interp.values))
# Compute R^2 for the scatter of interpolated Photospec CCI vs filtered MCD19 CCI
if np.sum(valid_mask) > 1:
    slope, intercept, r_value, p_value, std_err = linregress(
        photospec_cci_interp.values[valid_mask],
        MCD19_QC1_snowfilt['CCI'].values[valid_mask]
    )
    print(f"MCD19 QC1 QC scheme - Fit parameters (interp): slope={slope:.3f}, intercept={intercept:.3f}, R^2={r_value**2:.3f}")
else:
    print("Not enough valid data for regression (interp).")


plt.figure(figsize=(3.55, 3.3))
plt.scatter(photospec_cci_interp.values[valid_mask], MCD19_QC1_snowfilt['CCI'].values[valid_mask], color='red', alpha=0.7)
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
plt.title(QC1_descr)
plt.tight_layout()
plt.show()


#%%


# Only compare where both are valid (not nan)
valid_mask = (~np.isnan(MCD19_QC2_snowfilt['CCI'].values)) & (~np.isnan(photospec_cci_interp.values))
# Compute R^2 for the scatter of interpolated Photospec CCI vs filtered MCD19 CCI
if np.sum(valid_mask) > 1:
    slope, intercept, r_value, p_value, std_err = linregress(
        photospec_cci_interp.values[valid_mask],
        MCD19_QC2_snowfilt['CCI'].values[valid_mask]
    )
    print(f"MCD19 QC2 QC scheme - Fit parameters (interp): slope={slope:.3f}, intercept={intercept:.3f}, R^2={r_value**2:.3f}")
else:
    print("Not enough valid data for regression (interp).")


# Only compare where both are valid (not nan)
valid_mask = (~np.isnan(MCD19_QC2_snowfilt['CCI'].values)) & (~np.isnan(photospec_cci_interp.values))
plt.figure(figsize=(3.55, 3.3))
plt.scatter(photospec_cci_interp.values[valid_mask], MCD19_QC2_snowfilt['CCI'].values[valid_mask], color='blue', alpha=0.7, label='Filtered MCD19 vs Photospec (interp)')
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
plt.xlabel('Photospec CCI')
plt.ylabel('MCD19 CCI')
plt.title(QC2_descr)
plt.ylim(mcd19_cci_min - (mcd19_cci_range*0.05), mcd19_cci_max + (mcd19_cci_range*0.05))
plt.tight_layout()
plt.show()


# %%

# Only compare where both are valid (not nan)
valid_mask = (~np.isnan(MCD19_QC3_snowfilt['CCI'].values)) & (~np.isnan(photospec_cci_interp.values))

# Compute R^2 for the scatter of interpolated Photospec CCI vs filtered MCD19 CCI
if np.sum(valid_mask) > 1:
    slope, intercept, r_value, p_value, std_err = linregress(
        photospec_cci_interp.values[valid_mask],
        MCD19_QC3_snowfilt['CCI'].values[valid_mask]
    )
    print(f"MCD19 QC3 QC scheme - Fit parameters (interp): slope={slope:.3f}, intercept={intercept:.3f}, R^2={r_value**2:.3f}")
else:
    print("Not enough valid data for regression (interp).")


plt.figure(figsize=(3.55, 3.55))
plt.scatter(photospec_cci_interp.values[valid_mask], MCD19_QC3_snowfilt['CCI'].values[valid_mask], color='orange', alpha=0.7, label='Filtered MCD19 vs Photospec (interp)')
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
plt.title(QC3_descr)
plt.tight_layout()
plt.show()



# %%
