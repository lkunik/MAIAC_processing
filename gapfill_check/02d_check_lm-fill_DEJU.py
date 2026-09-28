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
site_name = 'DEJU'

MODIS_tile = 'h11v02'

# Compare AOD and adjacency masks:
# 1: CloudFree_AllAOD_AllAdj
# 2: CloudFree_LowAOD_AllAdj

# 3: CloudFree_AllAOD_NotSurround
# 4: CloudFree_LowAOD_NotSurround

# 5: CloudFree_AllAOD_ClearAdj
# 6: CloudFree_LowAOD_ClearAdj

QC_descr = "CloudFree_LowAOD_ClearAdj"


MCD19_dir_orig = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/pre-fill/QCfilt_{QC_descr}/{MODIS_tile}/'
MCD19_dir_hist_avg = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/pre-fill/QCfilt_{QC_descr}/{MODIS_tile}/hist_avg/'
MCD19_dir_outlierFilter = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/pre-fill_OutlierFilter/QCfilt_{QC_descr}/{MODIS_tile}/'
MCD19_dir_lmfill =f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/lm-fill/QCfilt_{QC_descr}/{MODIS_tile}/'


Photospec_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/data/PhotoSpec'
Photospec_file = os.path.join(Photospec_dir, f'PhotoSpec_{site_name}.csv')


MOD13_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/MODIS/NDVI_EVI_MOD13A2/01d'
MCD43_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/MODIS/Reflectance_MCD43A4/01d'

out_stats_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/output/CVMVC_QC_regression_stats/'
os.makedirs(out_stats_dir, exist_ok=True)

plot_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/plots/CVMVC_QC_regression_stats/'
os.makedirs(plot_dir, exist_ok=True)

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

DOY_composite_starts = np.arange(1, 366, 16)
composite_periods = range(len(DOY_composite_starts))  # 0-22

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



MCD19_orig_arr = []
MCD19_outlierFilt_arr = []
MCD19_lmfill_arr = []

# composite_year = 2015
for composite_year in composite_years:
    print(f"Loading MCD19 year {composite_year}...")

    composite_nn = 12
    for composite_nn in range(0, 23):
        print(f"  composite {composite_nn:02d}...")

        # List all files for this month portion
        # e.g. MCD19A1.h09v05.061_1km_16day_CV-MVC_QCfilt_CloudFree_AllAOD_AllAdj_2025-21.nc
        MCD19A1_orig_files = glob.glob(f'{MCD19_dir_orig}MCD19A1.{MODIS_tile}.*_{composite_year}-{composite_nn:02d}.nc')
        MCD19A1_outlierFilt_files = glob.glob(f'{MCD19_dir_outlierFilter}MCD19A1.{MODIS_tile}.*_{composite_year}-{composite_nn:02d}.nc')
        MCD19A1_lmfill_files = glob.glob(f'{MCD19_dir_lmfill}MCD19A1.{MODIS_tile}.*_{composite_year}-{composite_nn:02d}.nc')


        if len(MCD19A1_orig_files) > 0:
            MCD19A1_orig_file = MCD19A1_orig_files[0]  # Assuming there's only one file per month portion
            MCD19A1_orig_ds_curv = xr.open_dataset(MCD19A1_orig_file)
            MCD19A1_orig_ds_pt = clip_2dcurv_ds_to_point(MCD19A1_orig_ds_curv, site_lon, site_lat)
            MCD19_orig_arr.append(MCD19A1_orig_ds_pt)
        else:
            print(f"No files for {composite_year}-{composite_nn:02d} in {MCD19_dir_orig}")
            
        if len(MCD19A1_outlierFilt_files) > 0:
            MCD19A1_outlierFilt_file = MCD19A1_outlierFilt_files[0]  # Assuming there's only one file per month portion
            MCD19A1_outlierFilt_ds_curv = xr.open_dataset(MCD19A1_outlierFilt_file)
            MCD19A1_outlierFilt_ds_pt = clip_2dcurv_ds_to_point(MCD19A1_outlierFilt_ds_curv, site_lon, site_lat)
            MCD19_outlierFilt_arr.append(MCD19A1_outlierFilt_ds_pt)
        else:
            print(f"No files for {composite_year}-{composite_nn:02d} in {MCD19_dir_outlierFilter}")

        if len(MCD19A1_lmfill_files) > 0:
            MCD19A1_lmfill_file = MCD19A1_lmfill_files[0]  # Assuming there's only one file per month portion
            MCD19A1_lmfill_ds_curv = xr.open_dataset(MCD19A1_lmfill_file)
            MCD19A1_lmfill_ds_pt = clip_2dcurv_ds_to_point(MCD19A1_lmfill_ds_curv, site_lon, site_lat)
            MCD19_lmfill_arr.append(MCD19A1_lmfill_ds_pt)
        else:
            print(f"No files for {composite_year}-{composite_nn:02d} in {MCD19_dir_lmfill}")

MCD19_orig_pt = xr.concat(MCD19_orig_arr, dim='time')
MCD19_outlierFilt_pt = xr.concat(MCD19_outlierFilt_arr, dim='time')
MCD19_lmfill_pt = xr.concat(MCD19_lmfill_arr, dim='time')

all_times = np.union1d(np.union1d(MCD19_orig_pt.time.values, MCD19_outlierFilt_pt.time.values), MCD19_lmfill_pt.time.values)
MCD19_orig_pt = MCD19_orig_pt.reindex(time=all_times)
MCD19_outlierFilt_pt = MCD19_outlierFilt_pt.reindex(time=all_times)
MCD19_lmfill_pt = MCD19_lmfill_pt.reindex(time=all_times)



# ## Compute CCI from MCD19
MCD19_orig_pt = MCD19_orig_pt.assign(CCI = (MCD19_orig_pt.Sur_refl11 - MCD19_orig_pt.Sur_refl1) / (MCD19_orig_pt.Sur_refl11 + MCD19_orig_pt.Sur_refl1))
MCD19_outlierFilt_pt = MCD19_outlierFilt_pt.assign(CCI = (MCD19_outlierFilt_pt.Sur_refl11 - MCD19_outlierFilt_pt.Sur_refl1) / (MCD19_outlierFilt_pt.Sur_refl11 + MCD19_outlierFilt_pt.Sur_refl1))
MCD19_lmfill_pt = MCD19_lmfill_pt.assign(CCI = (MCD19_lmfill_pt.Sur_refl11 - MCD19_lmfill_pt.Sur_refl1) / (MCD19_lmfill_pt.Sur_refl11 + MCD19_lmfill_pt.Sur_refl1))

# ## Compute NDVI from MCD19
MCD19_orig_pt = MCD19_orig_pt.assign(NDVI = (MCD19_orig_pt.Sur_refl2 - MCD19_orig_pt.Sur_refl1) / (MCD19_orig_pt.Sur_refl2 + MCD19_orig_pt.Sur_refl1))
MCD19_outlierFilt_pt = MCD19_outlierFilt_pt.assign(NDVI = (MCD19_outlierFilt_pt.Sur_refl2 - MCD19_outlierFilt_pt.Sur_refl1) / (MCD19_outlierFilt_pt.Sur_refl2 + MCD19_outlierFilt_pt.Sur_refl1))
MCD19_lmfill_pt = MCD19_lmfill_pt.assign(NDVI = (MCD19_lmfill_pt.Sur_refl2 - MCD19_lmfill_pt.Sur_refl1) / (MCD19_lmfill_pt.Sur_refl2 + MCD19_lmfill_pt.Sur_refl1))

#%%

MCD19_histAvg_arr = []

for composite_nn in range(0, 23):
    print(f"HIST AVG - composite {composite_nn:02d}...")

    # List all files for this month portion
    # e.g. MCD19A1.h09v05.061_1km_16day_CV-MVC_QCfilt_CloudFree_AllAOD_AllAdj_2025-21.nc
    MCD19A1_histAvg_files = glob.glob(f'{MCD19_dir_hist_avg}MCD19A1.{MODIS_tile}.*_hist_avg_{composite_nn:02d}.nc')


    if len(MCD19A1_histAvg_files) > 0:
        MCD19A1_histAvg_file = MCD19A1_histAvg_files[0]  # Assuming there's only one file per month portion
        MCD19A1_histAvg_ds_curv = xr.open_dataset(MCD19A1_histAvg_file)
        MCD19A1_histAvg_ds_pt = clip_2dcurv_ds_to_point(MCD19A1_histAvg_ds_curv, site_lon, site_lat).expand_dims(doy = [DOY_composite_starts[composite_nn]])
        MCD19_histAvg_arr.append(MCD19A1_histAvg_ds_pt)
    else:
        print(f"No files for {composite_year}-{composite_nn:02d} in {MCD19_dir_hist_avg}")
        


MCD19_histAvg_pt = xr.concat(MCD19_histAvg_arr, dim='doy')


# ## Compute CCI from MCD19
MCD19_histAvg_pt = MCD19_histAvg_pt.assign(CCI = (MCD19_histAvg_pt.Sur_refl11 - MCD19_histAvg_pt.Sur_refl1) / (MCD19_histAvg_pt.Sur_refl11 + MCD19_histAvg_pt.Sur_refl1))

# ## Compute NDVI from MCD19
MCD19_histAvg_pt = MCD19_histAvg_pt.assign(NDVI = (MCD19_histAvg_pt.Sur_refl2 - MCD19_histAvg_pt.Sur_refl1) / (MCD19_histAvg_pt.Sur_refl2 + MCD19_histAvg_pt.Sur_refl1))



#%%

# # If already loaded, just read from temp
# temp_dat_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/data/temp'
# MCD19_orig_temp_file = os.path.join(temp_dat_dir, f'MCD19_QCfilt_{QC_descr}_{site_name}.nc')
# MCD19_outlierFilt_temp_file = os.path.join(temp_dat_dir, f'MCD19_QCfilt_{QC_descr}outlierFilt_{site_name}.nc')

# # MCD19_orig_pt = xr.open_dataset(MCD19_orig_temp_file)
# MCD19_outlierFilt_pt = xr.open_dataset(MCD19_outlierFilt_temp_file)



# #### SPECIFIC TO THIS VERSION HERE - re-order if necessary
# all_times = np.union1d(MCD19_orig_pt.time.values, MCD19_outlierFilt_pt.time.values)
# MCD19_orig_pt = MCD19_orig_pt.reindex(time=all_times)
# MCD19_outlierFilt_pt = MCD19_outlierFilt_pt.reindex(time=all_times)



# # # save data to temp
# temp_dat_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/data/temp'
# MCD19_orig_temp_file = os.path.join(temp_dat_dir, f'MCD19_QCfilt_HighVZA-MVC_{QC1_descr}_{site_name}.nc')
# # MCD19_outlierFilt_temp_file = os.path.join(temp_dat_dir, f'MCD19_QCfilt_{QC2_descr}_{site_name}.nc')
# # MCD19_QC3_temp_file = os.path.join(temp_dat_dir, f'MCD19_QCfilt_{QC3_descr}_{site_name}.nc')
# # MCD19_QC4_temp_file = os.path.join(temp_dat_dir, f'MCD19_QCfilt_{QC4_descr}_{site_name}.nc')
# # # MOD13_temp_file = os.path.join(temp_dat_dir, f'MOD13_pt_{site_name}.nc')
# # # MCD43_temp_file = os.path.join(temp_dat_dir, f'MCD43_pt_{site_name}.nc')
# MCD19_orig_pt.to_netcdf(MCD19_orig_temp_file)
# MCD19_outlierFilt_pt.to_netcdf(MCD19_outlierFilt_temp_file)
# MCD19_QC3_pt.to_netcdf(MCD19_QC3_temp_file)
# MCD19_QC4_pt.to_netcdf(MCD19_QC4_temp_file)
# # MOD13_pt.to_netcdf(MOD13_temp_file)
# # MCD43_pt.to_netcdf(MCD43_temp_file)

#%%

Photospec_df = pd.read_csv(Photospec_file)
print("Photospec_df columns:", Photospec_df.columns.tolist())

# Remove any data from Photospec_df where NDVI is below 0.3
# Photospec_df = Photospec_df[Photospec_df['NDVI'] >= 0.3].reset_index(drop=True)
# Convert Photospec timestamps from "M/D/YY HH:MM" format to datetime64
Photospec_times = pd.to_datetime(Photospec_df['Time'], format='%m/%d/%y %H:%M')

# Select MCD19 data within the time range of Photospec data
MCD19_orig_pt = MCD19_orig_pt.sel(time=slice(Photospec_times.iloc[0], Photospec_times.iloc[-1]))
MCD19_outlierFilt_pt = MCD19_outlierFilt_pt.sel(time=slice(Photospec_times.iloc[0], Photospec_times.iloc[-1]))


MCD19_orig_times = np.array([np.datetime64(t) for t in MCD19_orig_pt['time'].values])
MCD19_outlierFilt_times = np.array([np.datetime64(t) for t in MCD19_outlierFilt_pt['time'].values])
MCD19_lmfill_times = np.array([np.datetime64(t) for t in MCD19_lmfill_pt['time'].values])

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


# Calculate day of year for MCD19_CloudFree_times and Photospec_times_filtered
mcd19_doy = pd.to_datetime(MCD19_orig_times).dayofyear
photospec_doy = Photospec_times_filtered.dt.dayofyear
# mcd19_qc4_doy = pd.to_datetime(MCD19_QC4_times).dayofyear

# Sort data by day of year so you can plot as lines without weird zigzags
mcd19_doy_sorted = mcd19_doy[np.argsort(mcd19_doy)]
MCD19_orig_ndvi_sorted = MCD19_orig_pt['NDVI'].values[np.argsort(mcd19_doy)]
MCD19_outlierFilt_ndvi_sorted = MCD19_outlierFilt_pt['NDVI'].values[np.argsort(mcd19_doy)]
MCD19_lmfill_ndvi_sorted = MCD19_lmfill_pt['NDVI'].values[np.argsort(mcd19_doy)]

#%%
# Set global matplotlib font sizes for axes and legends
plt.rcParams.update({
   'axes.labelsize': 15,
   'axes.titlesize': 15,
   'xtick.labelsize': 15,
   'ytick.labelsize': 15,
   'legend.fontsize': 13
})


MCD19_orig_color = '#BF2665'
MCD19_outlierFilt_color = '#DB830B' # 
MCD19_histAvg_color = "#230AA1"
MCD19_lmfill_color = '#BB27F5'
# MCD19_QC3_color = '#BF2665' # #C131BC
# MCD19_QC4_color = '#DB830B'
MOD13_color = '#0DA10A'
MCD43_color = '#B5C230'

winter_periods = [
    # Ground snow cover periods
    (pd.Timestamp('2018-10-28'), pd.Timestamp('2019-04-15')),
    (pd.Timestamp('2019-10-02'), pd.Timestamp('2020-05-05')),
    (pd.Timestamp('2020-10-13'), pd.Timestamp('2021-05-05')),
    (pd.Timestamp('2021-09-20'), pd.Timestamp('2022-05-20'))
]
############################################
### QC ANALYSIS DELIVERABLE 1 - CCI time series 
############################################

#%%
# Calculate day of year for MCD19_CloudFree_times and Photospec_times_filtered
mcd19_doy = pd.to_datetime(MCD19_orig_times).dayofyear
photospec_doy = Photospec_times_filtered.dt.dayofyear

cci_arrays = [
    MCD19_orig_pt['CCI'].values,
    MCD19_outlierFilt_pt['CCI'].values,
    MCD19_lmfill_pt['CCI'].values,
]
mcd19_cci_min = np.nanmin(np.concatenate(cci_arrays))
mcd19_cci_max = np.nanmax(np.concatenate(cci_arrays))
mcd19_cci_range = mcd19_cci_max - mcd19_cci_min

fig, ax = plt.subplots(figsize=(7, 4))

for start, end in winter_periods:
    ax.axvspan(start, end, color='lightblue', alpha=0.5)
ax.scatter(Photospec_times_filtered, Photospec_df_filtered['CCI'], color='black', s=10, alpha=0.7, label='Photospec')
ax2 = ax.twinx()

ax2.plot(MCD19_orig_times, MCD19_orig_pt['CCI'].values, color=MCD19_orig_color, marker = 'o', markersize=6, linewidth=3, alpha=1, markeredgecolor='black', markeredgewidth=0.5)
ax2.plot(MCD19_lmfill_times, MCD19_lmfill_pt['CCI'].values, color=MCD19_lmfill_color, marker = 'o', markersize=6, linewidth=3, alpha=1, markeredgecolor='black', markeredgewidth=0.5)
ax2.plot(MCD19_outlierFilt_times, MCD19_outlierFilt_pt['CCI'].values, color=MCD19_outlierFilt_color, marker = 'o', markersize=6, linewidth=3, alpha=1, markeredgecolor='black', markeredgewidth=0.5)

ax2.set_ylim(mcd19_cci_min - (mcd19_cci_range*0.05), mcd19_cci_max + (mcd19_cci_range*0.05))
ax.yaxis.set_major_formatter(FFmt(lambda y, _: f'{y:.1f}'))
ax2.yaxis.set_major_formatter(FFmt(lambda y, _: f'{y:.1f}'))
ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax2.yaxis.set_major_locator(MultipleLocator(0.1))
ax.set_xlim(Photospec_times_filtered.min() - pd.Timedelta(days=10), Photospec_times_filtered.max() + pd.Timedelta(days=10))
ax.xaxis.set_major_locator(MonthLocator(bymonth=[7, 10], interval=1))
ax.xaxis.set_major_locator(MonthLocator(bymonth=[1, 4, 7, 10]))
ax.xaxis.set_major_formatter(DateFormatter('%b\n%Y'))
ax.set_title('')
ax.set_ylabel('CCI')

# Combine legends using Patch
handles = [
    Patch(color = 'lightblue', label=f'Estimated snow cover period', alpha=0.3),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label='PhotoSpec'),
    plt.Line2D([0], [0], color=MCD19_orig_color, linewidth=2, label=f'MCD19 (pre-fill)'),
    plt.Line2D([0], [0], color=MCD19_outlierFilt_color, linewidth=2, label=f'MCD19 (outlier filter)'),
    plt.Line2D([0], [0], color=MCD19_lmfill_color, linewidth=2, label=f'MCD19 (lm-fill)')
]
ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.3), fontsize=10, ncol=1)
plt.subplots_adjust(bottom=0.35)

plt.tight_layout()
plt.show()

# plot_filename = os.path.join(plot_dir, f'MCD19_v_PhotoSpec_timeseries_{out_stats_descr}_CCI_{site_name}.png')
# plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
# plt.close()

#%%

############################################
### QC ANALYSIS DELIVERABLE 1b - CCI time series with histAvg overlay
############################################

# Extract day of year from MCD19_orig_times for alignment
mcd19_orig_doy = pd.to_datetime(MCD19_orig_times).dayofyear

# Create a mapping from doy to MCD19_histAvg CCI values
histAvg_cci_by_doy = {}
for doy_val in MCD19_histAvg_pt.doy.values:
    cci_val = MCD19_histAvg_pt.sel(doy=doy_val)['CCI'].values
    histAvg_cci_by_doy[doy_val] = cci_val

# Map histAvg CCI values to MCD19_orig_times by matching day of year
histAvg_cci_aligned = np.array([histAvg_cci_by_doy.get(doy, np.nan) for doy in mcd19_orig_doy])

# Calculate y-axis limits including histAvg data
cci_arrays_with_histAvg = [
    MCD19_orig_pt['CCI'].values,
    MCD19_outlierFilt_pt['CCI'].values,
    histAvg_cci_aligned
]
mcd19_cci_min_with_histAvg = np.nanmin(np.concatenate(cci_arrays_with_histAvg))
mcd19_cci_max_with_histAvg = np.nanmax(np.concatenate(cci_arrays_with_histAvg))
mcd19_cci_range_with_histAvg = mcd19_cci_max_with_histAvg - mcd19_cci_min_with_histAvg

fig, ax = plt.subplots(figsize=(7, 4))

for start, end in winter_periods:
    ax.axvspan(start, end, color='lightblue', alpha=0.5)
ax.scatter(Photospec_times_filtered, Photospec_df_filtered['CCI'], color='black', s=10, alpha=0.7, label='Photospec')
ax2 = ax.twinx()

ax2.plot(MCD19_orig_times, MCD19_orig_pt['CCI'].values, color=MCD19_orig_color, marker='o', markersize=6, linewidth=3, alpha=1, markeredgecolor='black', markeredgewidth=0.5)
ax2.plot(MCD19_lmfill_times, MCD19_lmfill_pt['CCI'].values, color=MCD19_lmfill_color, marker='o', markersize=6, linewidth=3, alpha=1, markeredgecolor='black', markeredgewidth=0.5)
ax2.plot(MCD19_outlierFilt_times, MCD19_outlierFilt_pt['CCI'].values, color=MCD19_outlierFilt_color, marker='o', markersize=6, linewidth=3, alpha=1, markeredgecolor='black', markeredgewidth=0.5)
ax2.plot(MCD19_orig_times, histAvg_cci_aligned, color=MCD19_histAvg_color, linewidth=3, linestyle='--', alpha=0.8)

ax2.set_ylim(mcd19_cci_min_with_histAvg - (mcd19_cci_range_with_histAvg*0.05), mcd19_cci_max_with_histAvg + (mcd19_cci_range_with_histAvg*0.05))
ax.yaxis.set_major_formatter(FFmt(lambda y, _: f'{y:.1f}'))
ax2.yaxis.set_major_formatter(FFmt(lambda y, _: f'{y:.1f}'))
ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax2.yaxis.set_major_locator(MultipleLocator(0.1))
ax.set_xlim(Photospec_times_filtered.min() - pd.Timedelta(days=10), Photospec_times_filtered.max() + pd.Timedelta(days=10))
ax.xaxis.set_major_locator(MonthLocator(bymonth=[1, 4, 7, 10]))
ax.xaxis.set_major_formatter(DateFormatter('%b\n%Y'))
ax.set_title('')
ax.set_ylabel('CCI')

# Combine legends using Patch
handles = [
    Patch(color='lightblue', label=f'Estimated snow cover period', alpha=0.3),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label='PhotoSpec'),
    plt.Line2D([0], [0], color=MCD19_orig_color, linewidth=2, label=f'MCD19 (pre-fill)'),
    plt.Line2D([0], [0], color=MCD19_outlierFilt_color, linewidth=2, label=f'MCD19 (outlier filter)'),
    plt.Line2D([0], [0], color=MCD19_histAvg_color, marker='s', markersize=8, linewidth=2, label=f'MCD19 (hist avg)'),
    plt.Line2D([0], [0], color=MCD19_lmfill_color, linewidth=2, label=f'MCD19 (lm-fill)')]
ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.3), fontsize=10, ncol=1)
plt.subplots_adjust(bottom=0.35)

plt.tight_layout()
plt.show()

#%%

############################################
### QC ANALYSIS DELIVERABLE 2 - NDVI time series 
############################################


# Calculate day of year for MCD19_CloudFree_times and Photospec_times_filtered


ndvi_arrays = [
    MCD19_orig_pt['NDVI'].values,
    MCD19_outlierFilt_pt['NDVI'].values,
    MCD19_lmfill_pt['NDVI'].values
]
mcd19_ndvi_min = np.nanmin(np.concatenate(ndvi_arrays))
mcd19_ndvi_max = np.nanmax(np.concatenate(ndvi_arrays))
mcd19_ndvi_range = mcd19_ndvi_max - mcd19_ndvi_min


fig, ax = plt.subplots(figsize=(7, 4.5))

for start, end in winter_periods:
    ax.axvspan(start, end, color='lightblue', alpha=0.5)
ax.scatter(Photospec_times_filtered, Photospec_df_filtered['NDVI'], color='black', s=10, alpha=0.7, label='Photospec')
ax2 = ax.twinx()

ax2.plot(MCD19_orig_times, MCD19_orig_pt['NDVI'].values, color=MCD19_orig_color, marker = 'o', markersize=6, linewidth=3, alpha=1, markeredgecolor='black', markeredgewidth=0.5)
ax2.plot(MCD19_lmfill_times, MCD19_lmfill_pt['NDVI'].values, color=MCD19_lmfill_color, marker = 'o', markersize=6, linewidth=3, alpha=1, markeredgecolor='black', markeredgewidth=0.5)
ax2.plot(MCD19_outlierFilt_times, MCD19_outlierFilt_pt['NDVI'].values, color=MCD19_outlierFilt_color, marker = 'o', markersize=6, linewidth=3, alpha=1, markeredgecolor='black', markeredgewidth=0.5)

ax2.set_ylim(mcd19_ndvi_min - (mcd19_ndvi_range*0.05), mcd19_ndvi_max + (mcd19_ndvi_range*0.05))
ax2.plot(MCD19_outlierFilt_times, MCD19_outlierFilt_pt['NDVI'].values, color=MCD19_outlierFilt_color, marker = 'o', markersize=6, linewidth=3, alpha=1, markeredgecolor='black', markeredgewidth=0.5)
ax.yaxis.set_major_formatter(FFmt(lambda y, _: f'{y:.1f}'))
ax2.yaxis.set_major_formatter(FFmt(lambda y, _: f'{y:.1f}'))
ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax2.yaxis.set_major_locator(MultipleLocator(0.1))

ax.set_xlim(Photospec_times_filtered.min() - pd.Timedelta(days=10), Photospec_times_filtered.max() + pd.Timedelta(days=10))
ax.xaxis.set_major_locator(MonthLocator(bymonth=[7, 10], interval=1))
ax.xaxis.set_major_locator(MonthLocator(bymonth=[1, 4, 7, 10]))
ax.xaxis.set_major_formatter(DateFormatter('%b\n%Y'))
ax.set_title('')
ax.set_ylabel('NDVI')

# Combine legends using Patch
handles = [
    Patch(color = 'lightblue', label=f'Estimated snow cover period', alpha=0.3),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label='PhotoSpec'),
    plt.Line2D([0], [0], color=MCD19_orig_color, linewidth=2, label=f'MCD19 (pre-fill)'),
    plt.Line2D([0], [0], color=MCD19_outlierFilt_color, linewidth=2, label=f'MCD19 (outlier filter)'),
    plt.Line2D([0], [0], color=MCD19_lmfill_color, linewidth=2, label=f'MCD19 (lm-fill)')
]
ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.3), fontsize=10, ncol=1)
plt.subplots_adjust(bottom=0.35)

plt.tight_layout()
plt.show()





# %%


############################################
### QC ANALYSIS DELIVERABLE 2b - NDVI time series with histAvg overlay
############################################

# Extract day of year from MCD19_orig_times for alignment
mcd19_orig_doy = pd.to_datetime(MCD19_orig_times).dayofyear

# Create a mapping from doy to MCD19_histAvg NDVI values
histAvg_ndvi_by_doy = {}
for doy_val in MCD19_histAvg_pt.doy.values:
    ndvi_val = MCD19_histAvg_pt.sel(doy=doy_val)['NDVI'].values
    histAvg_ndvi_by_doy[doy_val] = ndvi_val

# Map histAvg NDVI values to MCD19_orig_times by matching day of year
histAvg_ndvi_aligned = np.array([histAvg_ndvi_by_doy.get(doy, np.nan) for doy in mcd19_orig_doy])

# Calculate y-axis limits including histAvg data
ndvi_arrays_with_histAvg = [
    MCD19_orig_pt['NDVI'].values,
    MCD19_outlierFilt_pt['NDVI'].values,
    histAvg_ndvi_aligned
]
mcd19_ndvi_min_with_histAvg = np.nanmin(np.concatenate(ndvi_arrays_with_histAvg))
mcd19_ndvi_max_with_histAvg = np.nanmax(np.concatenate(ndvi_arrays_with_histAvg))
mcd19_ndvi_range_with_histAvg = mcd19_ndvi_max_with_histAvg - mcd19_ndvi_min_with_histAvg

fig, ax = plt.subplots(figsize=(7, 4))

for start, end in winter_periods:
    ax.axvspan(start, end, color='lightblue', alpha=0.5)
ax.scatter(Photospec_times_filtered, Photospec_df_filtered['NDVI'], color='black', s=10, alpha=0.7, label='Photospec')
ax2 = ax.twinx()

ax2.plot(MCD19_orig_times, MCD19_orig_pt['NDVI'].values, color=MCD19_orig_color, marker='o', markersize=6, linewidth=3, alpha=1, markeredgecolor='black', markeredgewidth=0.5)
ax2.plot(MCD19_lmfill_times, MCD19_lmfill_pt['NDVI'].values, color=MCD19_lmfill_color, marker='o', markersize=6, linewidth=3, alpha=1, markeredgecolor='black', markeredgewidth=0.5)
ax2.plot(MCD19_outlierFilt_times, MCD19_outlierFilt_pt['NDVI'].values, color=MCD19_outlierFilt_color, marker='o', markersize=6, linewidth=3, alpha=1, markeredgecolor='black', markeredgewidth=0.5)
ax2.plot(MCD19_orig_times, histAvg_ndvi_aligned, color=MCD19_histAvg_color, linewidth=3, linestyle='--', alpha=0.8)

ax2.set_ylim(mcd19_ndvi_min_with_histAvg - (mcd19_ndvi_range_with_histAvg*0.05), mcd19_ndvi_max_with_histAvg + (mcd19_ndvi_range_with_histAvg*0.05))
ax.yaxis.set_major_formatter(FFmt(lambda y, _: f'{y:.1f}'))
ax2.yaxis.set_major_formatter(FFmt(lambda y, _: f'{y:.1f}'))
ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax2.yaxis.set_major_locator(MultipleLocator(0.1))
ax.set_xlim(Photospec_times_filtered.min() - pd.Timedelta(days=10), Photospec_times_filtered.max() + pd.Timedelta(days=10))
ax.xaxis.set_major_locator(MonthLocator(bymonth=[1, 4, 7, 10]))
ax.xaxis.set_major_formatter(DateFormatter('%b\n%Y'))
ax.set_title('')
ax.set_ylabel('NDVI')

# Combine legends using Patch
handles = [
    Patch(color='lightblue', label=f'Estimated snow cover period', alpha=0.3),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label='PhotoSpec'),
    plt.Line2D([0], [0], color=MCD19_orig_color, linewidth=2, label=f'MCD19 (pre-fill)'),
    plt.Line2D([0], [0], color=MCD19_outlierFilt_color, linewidth=2, label=f'MCD19 (outlier filter)'),
    plt.Line2D([0], [0], color=MCD19_histAvg_color, marker='s', markersize=8, linewidth=2, label=f'MCD19 (hist avg)'),
    plt.Line2D([0], [0], color=MCD19_lmfill_color, linewidth=2, label=f'MCD19 (lm-fill)')
]
ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.3), fontsize=10, ncol=1)
plt.subplots_adjust(bottom=0.35)

plt.tight_layout()
plt.show()


# %%
