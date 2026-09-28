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
MODIS_tile = 'h11v03'

# Compare AOD and adjacency masks:
# 1: CloudFree_AllAOD_AllAdj
# 2: CloudFree_LowAOD_AllAdj

# 3: CloudFree_AllAOD_NotSurround
# 4: CloudFree_LowAOD_NotSurround

# 5: CloudFree_AllAOD_ClearAdj
# 6: CloudFree_LowAOD_ClearAdj

QC1_descr = "CloudFree_LowAOD_ClearAdj"
QC2_descr = "CloudFree_LowAOD_ClearAdj"
# QC3_descr = "CloudFree_AllAOD_AllAdj"
# QC4_descr = "CloudFree_AllAOD_ClearAdj"

QC1_legend_descr = "40-70° View Angle"
QC2_legend_descr = "All View Angles"

dat_pt_basedir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/data/QC_PhotoSpec_sites_backup/'
dat_pt_basedir_QC1 = os.path.join(dat_pt_basedir, f'HighVZA-MVC/pre-fill/QCfilt_{QC1_descr}')
dat_pt_basedir_QC2 = os.path.join(dat_pt_basedir, f'CV-MVC/pre-fill/VZA0-90/QCfilt_{QC2_descr}')

MCD19_QC1_file = os.path.join(dat_pt_basedir_QC1, f'MCD19_QCfilt_{QC1_descr}_{site_name}.nc')
MCD19_QC2_file = os.path.join(dat_pt_basedir_QC2, f'MCD19_QCfilt_{QC2_descr}_{site_name}.nc')
# MCD19_QC3_file = os.path.join(dat_pt_basedir_QC3, f'MCD19_QCfilt_{QC3_descr}_{site_name}.nc')
# MCD19_QC4_file = os.path.join(dat_pt_basedir_QC4, f'MCD19_QCfilt_{QC4_descr}_{site_name}.nc')

Photospec_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/data/PhotoSpec'
Photospec_file = os.path.join(Photospec_dir, f'PhotoSpec_{site_name}.csv')

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

#########################################
# Begin main
#########################################
#%%
# def main():

# mark start time to keep track of elapsed
start_total = time.time()

MCD19_QC1_pt = xr.open_dataset(MCD19_QC1_file)
MCD19_QC2_pt = xr.open_dataset(MCD19_QC2_file)
# MCD19_QC3_pt = xr.open_dataset(MCD19_QC3_file)
# MCD19_QC4_pt = xr.open_dataset(MCD19_QC4_file)


#### SPECIFIC TO THIS VERSION HERE - re-order if necessary
all_times = np.union1d(MCD19_QC1_pt.time.values, MCD19_QC2_pt.time.values)
MCD19_QC1_pt = MCD19_QC1_pt.reindex(time=all_times)
MCD19_QC2_pt = MCD19_QC2_pt.reindex(time=all_times)


#%%

Photospec_df = pd.read_csv(Photospec_file)
print("Photospec_df columns:", Photospec_df.columns.tolist())


# Print the time range in the 'Time' column of Photospec_df
print(f"Photospec time range: {Photospec_df['Time'].iloc[0]} to {Photospec_df['Time'].iloc[-1]}")


# Remove any data from Photospec_df where NDVI is below 0.4
Photospec_df = Photospec_df[Photospec_df['NDVI'] >= 0.4].reset_index(drop=True)
# Convert Photospec timestamps from "M/D/YY HH:MM" format to datetime64
Photospec_times = pd.to_datetime(Photospec_df['Time'], format='%m/%d/%y %H:%M')

# Select MCD19 data within the time range of Photospec data
MCD19_QC1_pt = MCD19_QC1_pt.sel(time=slice(Photospec_times.iloc[0], Photospec_times.iloc[-1]))
MCD19_QC2_pt = MCD19_QC2_pt.sel(time=slice(Photospec_times.iloc[0], Photospec_times.iloc[-1]))
# MCD19_QC3_pt = MCD19_QC3_pt.sel(time=slice(Photospec_times.iloc[0], Photospec_times.iloc[-1]))
# MCD19_QC4_pt = MCD19_QC4_pt.sel(time=slice(Photospec_times.iloc[0], Photospec_times.iloc[-1]))

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
# MCD19_QC4_times = np.array([np.datetime64(t) for t in MCD19_QC4_pt['time'].values])

#%%

# Set global matplotlib font sizes for axes and legends
plt.rcParams.update({
   'axes.labelsize': 15,
   'axes.titlesize': 15,
   'xtick.labelsize': 15,
   'ytick.labelsize': 15,
   'legend.fontsize': 13
})



MCD19_QC1_color = '#BF2665'
MCD19_QC2_color = '#DB830B' # 
# MCD19_QC3_color = '#BF2665' # #C131BC
# MCD19_QC4_color = '#DB830B'
MOD13_color = '#0DA10A'
MCD43_color = '#B5C230'


# Plot shaded regions for winter periods
winter_periods = [
    (pd.Timestamp('2018-10-30'), pd.Timestamp('2019-04-25')),
    (pd.Timestamp('2019-10-26'), pd.Timestamp('2020-05-05')),
    (pd.Timestamp('2020-10-15'), pd.Timestamp('2021-04-20')),
    (pd.Timestamp('2021-11-11'), pd.Timestamp('2022-05-10')),
    (pd.Timestamp('2022-11-02'), pd.Timestamp('2023-04-30'))
]



############################################
### QC ANALYSIS DELIVERABLE 1 - CCI time series 
############################################
#%%

# Calculate day of year for MCD19_CloudFree_times and Photospec_times_filtered
mcd19_doy = pd.to_datetime(MCD19_QC1_times).dayofyear
photospec_doy = Photospec_times_filtered.dt.dayofyear

nobs_arrays = [
    MCD19_QC1_pt['valid_obs_count'].values,
    MCD19_QC2_pt['valid_obs_count'].values,
    # MCD19_QC3_pt['valid_obs_count'].values,
    # MCD19_QC4_pt['valid_obs_count'].values
]
mcd19_nobs_min = np.nanmin(np.concatenate(nobs_arrays))
mcd19_nobs_max = np.nanmax(np.concatenate(nobs_arrays))
mcd19_nobs_range = mcd19_nobs_max - mcd19_nobs_min


# fig, ax = plt.subplots(figsize=(7, 5))

# for start, end in winter_periods:
#     ax.axvspan(start, end, color='lightblue', alpha=0.3)

# # ax.plot(MCD19_QC1_times, MCD19_QC1_pt['valid_obs_count'].values, color=MCD19_QC1_color, linewidth=3, alpha=1, label=QC1_descr)
# # ax.plot(MCD19_QC2_times, MCD19_QC2_pt['valid_obs_count'].values, color=MCD19_QC2_color, linewidth=3, alpha=1, label=QC2_descr)
# # ax.plot(MCD19_QC3_times, MCD19_QC3_pt['valid_obs_count'].values, color=MCD19_QC3_color, linewidth=3, alpha=1, label=QC3_descr)
# # ax.plot(MCD19_QC4_times, MCD19_QC4_pt['valid_obs_count'].values, color=MCD19_QC4_color, linewidth=3, alpha=1, label=QC4_descr)

# # ax.plot(MCD19_QC1_times, MCD19_QC1_pt['valid_obs_count'].values, color=MCD19_QC1_color, linewidth=3, alpha=1, label=QC1_descr)
# # ax.fill_between(MCD19_QC1_times, MCD19_QC1_pt['valid_obs_count'].values, -5, color=MCD19_QC1_color, alpha=1)

# # ax.plot(MCD19_QC3_times, MCD19_QC3_pt['valid_obs_count'].values, color=MCD19_QC3_color, linewidth=3, alpha=1, label=QC3_descr)
# ax.fill_between(MCD19_QC3_times, MCD19_QC3_pt['valid_obs_count'].values, -5, color=MCD19_QC3_color, alpha=1)
# # ax.plot(MCD19_QC4_times, MCD19_QC4_pt['valid_obs_count'].values, color=MCD19_QC4_color, linewidth=3, alpha=1, label=QC4_descr)
# ax.fill_between(MCD19_QC4_times, MCD19_QC4_pt['valid_obs_count'].values, -5, color=MCD19_QC4_color, alpha=1)
# # ax.plot(MCD19_QC2_times, MCD19_QC2_pt['valid_obs_count'].values, color=MCD19_QC2_color, linewidth=3, alpha=1, label=QC2_descr)
# ax.fill_between(MCD19_QC2_times, MCD19_QC2_pt['valid_obs_count'].values, -5, color=MCD19_QC2_color, alpha=1)



# ax.set_ylim(mcd19_nobs_min - (mcd19_nobs_range*0.05), mcd19_nobs_max + (mcd19_nobs_range*0.05))


# for start, end in winter_periods:
#     ax.axvline(x=start, color='navy', linestyle='--', alpha = 0.5)
#     ax.axvline(x=end, color='navy', linestyle='--', alpha = 0.5)

# # ax.yaxis.set_major_formatter(FFmt(lambda y, _: f'{y:.1f}'))
# ax.set_xlim(Photospec_times_filtered.min() - pd.Timedelta(days=10), Photospec_times_filtered.max() + pd.Timedelta(days=10))
# ax.xaxis.set_major_locator(MonthLocator(bymonth=[1]))
# ax.xaxis.set_major_formatter(DateFormatter('%b\n%Y'))
# ax.set_title('')

# # Combine legends using Patch
# handles = [
#     Patch(color = 'lightblue', label=f'Estimated snow cover timing', alpha=0.3),
#     plt.Line2D([0], [0], color=MCD19_QC1_color, linewidth=2, label=f'MCD19 ({QC1_descr})'),
#     plt.Line2D([0], [0], color=MCD19_QC2_color, linewidth=2, label=f'MCD19 ({QC2_descr})'),
#     plt.Line2D([0], [0], color=MCD19_QC3_color, linewidth=2, label=f'MCD19 ({QC3_descr})'),
#     plt.Line2D([0], [0], color=MCD19_QC4_color, linewidth=2, label=f'MCD19 ({QC4_descr})')
# ]
# ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.5), fontsize=10, ncol=1)
# plt.subplots_adjust(bottom=0.35)

# plt.tight_layout()
# plt.show()

# %%

#################################
### This is the "final" version
#################################



# Create normalized time axis: each composite is 16 days apart
composite_indices = np.arange(len(MCD19_QC1_pt.time.values))
normalized_days = composite_indices * 16

fig, ax = plt.subplots(figsize=(7, 5))

# Map actual dates to normalized axis for snow cover shading
actual_times = pd.to_datetime(MCD19_QC1_pt.time.values)


# Convert winter_periods to normalized axis
for start, end in winter_periods:
    # Find closest composite indices for this period
    start_idx = np.argmin(np.abs((actual_times - start).total_seconds()))
    end_idx = np.argmin(np.abs((actual_times - end).total_seconds()))
    start_norm = start_idx * 16
    end_norm = end_idx * 16    
    ax.axvspan(start_norm, end_norm, color='lightblue', alpha=0.5) # Plot on normalized axis


# Plot all QC options with uniform bar width
# ax.bar(normalized_days, MCD19_QC3_pt['valid_obs_count'].values, color=MCD19_QC3_color, alpha=1, width=16, label=QC3_descr)
# ax.bar(normalized_days, MCD19_QC4_pt['valid_obs_count'].values, color=MCD19_QC4_color, alpha=1, width=16, label=QC4_descr)
# ax.bar(normalized_days, MCD19_QC2_pt['valid_obs_count'].values, color=MCD19_QC2_color, alpha=1, width=16)
ax.bar(normalized_days, MCD19_QC2_pt['valid_obs_count'].values, color=MCD19_QC2_color, alpha=1, width=16)
ax.bar(normalized_days, MCD19_QC1_pt['valid_obs_count'].values, color=MCD19_QC1_color, alpha=1, width=16)

# # Convert winter_periods to normalized axis
# for start, end in winter_periods:
#     # Find closest composite indices for this period
#     start_idx = np.argmin(np.abs((actual_times - start).total_seconds()))
#     end_idx = np.argmin(np.abs((actual_times - end).total_seconds()))
#     start_norm = start_idx * 16
#     end_norm = end_idx * 16
#     # Plot on normalized axis
#     ax.axvline(x=start_norm, color='navy', linestyle='--', linewidth = 0.6, alpha=0.5)
#     ax.axvline(x=end_norm, color='navy', linestyle='--', linewidth = 0.6, alpha=0.5)


# Create x-axis labels: find which composites correspond to month starts
month_dates = pd.date_range(start='2018-01-01', end='2022-01-01', freq='1Y')
month_indices = [np.argmin(np.abs(actual_times - date)) for date in month_dates]
month_positions = [idx * 16 for idx in month_indices]
month_labels = [date.strftime('%b\n%Y') for date in month_dates]

ax.set_xticks(month_positions)
ax.set_xticklabels(month_labels, fontsize=13)
ax.set_ylabel('Number of Observations', fontsize=15)
ax.set_ylim(0, mcd19_nobs_max + (mcd19_nobs_range*0.15))


# Handle legend
handles = [
    Patch(color='lightblue', label='Estimated snow cover timing', alpha=0.3),
    # plt.Line2D([0], [0], color=MCD19_QC1_color, linewidth=6, label=QC1_descr),
    plt.Line2D([0], [0], color=MCD19_QC1_color, linewidth=6, label=QC1_legend_descr),
    plt.Line2D([0], [0], color=MCD19_QC2_color, linewidth=6, label=QC2_legend_descr),
    # plt.Line2D([0], [0], color=MCD19_QC3_color, linewidth=6, label=QC3_legend_descr),
    # plt.Line2D([0], [0], color=MCD19_QC4_color, linewidth=6, label=QC4_legend_descr)
]
ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.25), fontsize=11, ncol=1)

plt.tight_layout()
plt.show()
# %%
