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

QC_descr = "CloudFree_LowAOD_ClearAdj"

MCD19_legend_descr = "CV-MVC" # gets added in parentheses after "MCD19"


dat_pt_basedir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/data/QC_PhotoSpec_sites_backup/'
dat_pt_basedir_QC = os.path.join(dat_pt_basedir, f'CV-MVC/lm-fill/QCfilt_{QC_descr}')

MCD19_QC_file = os.path.join(dat_pt_basedir_QC, f'MCD19_QCfilt_{QC_descr}_{site_name}.nc')

Photospec_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/data/PhotoSpec'
Photospec_file = os.path.join(Photospec_dir, f'PhotoSpec_{site_name}.csv')

MOD13_dir = os.path.join(dat_pt_basedir, 'MOD13')
MOD13_file = os.path.join(MOD13_dir, f'MOD13_pt_{site_name}.nc')
MCD43_dir = os.path.join(dat_pt_basedir, 'MCD43')
MCD43_file = os.path.join(MCD43_dir, f'MCD43_pt_{site_name}.nc')

GCOM_dir = os.path.join(dat_pt_basedir, 'GCOM')
GCOM_file = os.path.join(GCOM_dir, f'GCOM_pt_{site_name}.nc')
MOD09_dir = os.path.join(dat_pt_basedir, 'MOD09')
MOD09_file = os.path.join(MOD09_dir, f'MOD09_pt_{site_name}.nc')

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

MCD19_QC_pt = xr.open_dataset(MCD19_QC_file)

MOD13_pt = xr.open_dataset(MOD13_file)
MCD43_pt = xr.open_dataset(MCD43_file)
GCOM_pt = xr.open_dataset(GCOM_file)
MOD09_pt = xr.open_dataset(MOD09_file)

Photospec_df = pd.read_csv(Photospec_file)
print("Photospec_df columns:", Photospec_df.columns.tolist())

# Remove any data from Photospec_df where NDVI is below 0.6
# Photospec_df = Photospec_df[Photospec_df['NDVI'] >= 0.6].reset_index(drop=True)
# Convert Photospec timestamps from "M/D/YY HH:MM" format to datetime64
Photospec_times = pd.to_datetime(Photospec_df['Time'], format='%m/%d/%y %H:%M')

# Select MCD19 data within the time range of Photospec data

# MCD19_QC_pt = MCD19_QC_pt.sel(time=slice(Photospec_times.iloc[0], Photospec_times.iloc[-1]))
MCD19_QC_pt = MCD19_QC_pt.sel(time=slice(
    Photospec_times.iloc[0] - pd.Timedelta(days=16),
    Photospec_times.iloc[-1] + pd.Timedelta(days=16),
))

MCD19_QC_times = np.array([np.datetime64(t) for t in MCD19_QC_pt['time'].values])

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

#%%
# Set global matplotlib font sizes for axes and legends
plt.rcParams.update({
   'axes.labelsize': 15,
   'axes.titlesize': 15,
   'xtick.labelsize': 15,
   'ytick.labelsize': 15,
   'legend.fontsize': 13
})


MCD19_QC_color = '#BF2665' #'#DB830B' 
GCOM_color = '#56B4E9'
MOD09_color = '#B288E9'
MOD13_color = '#0DA10A'
MCD43_color = '#B5C230'

# Plot shaded regions for winter periods
winter_periods = [
    # Ground snow cover periods
    (pd.Timestamp('2018-10-28'), pd.Timestamp('2019-04-15')),
    (pd.Timestamp('2019-10-02'), pd.Timestamp('2020-05-05')),
    (pd.Timestamp('2020-10-13'), pd.Timestamp('2021-05-05')),
    (pd.Timestamp('2021-09-20'), pd.Timestamp('2022-05-20'))
]

# %%


############################################
### QC ANALYSIS DELIVERABLE 1 - NDVI time series (v1 - one axis)
############################################


# Calculate day of year for MCD19_CloudFree_times and Photospec_times_filtered

ndvi_arrays = [
    MCD19_QC_pt['NDVI'].values,
    Photospec_df_filtered['NDVI']
]
mcd19_ndvi_min = np.nanmin(np.concatenate(ndvi_arrays))
mcd19_ndvi_max = np.nanmax(np.concatenate(ndvi_arrays))
mcd19_ndvi_range = mcd19_ndvi_max - mcd19_ndvi_min


fig, ax = plt.subplots(figsize=(7, 4.5))

for start, end in winter_periods:
    ax.axvspan(start, end, color='lightblue', alpha=0.5)
ax.scatter(Photospec_times_filtered, Photospec_df_filtered['NDVI'], color='black', s=10, alpha=0.7)

ax.scatter(MOD13_times, MOD13_pt['NDVI'].values, color=MOD13_color, s=35, alpha=1, marker='s', edgecolors='black', linewidths=0.5)
ax.scatter(MCD43_times, MCD43_pt['NDVI'].values, color=MCD43_color, s=35, alpha=1, marker='s', edgecolors='black', linewidths=0.5)

ax.scatter(MOD09_times, MOD09_pt['NDVI'].values, color=MOD09_color, s=60, alpha=1,  marker='^', edgecolors='black', linewidths=0.5, zorder = 2)
# ax.plot(GCOM_times, GCOM_pt['NDVI'].values, color=GCOM_color, marker = '^', markersize=9, linewidth=2.5, alpha=1, zorder = 2)
ax.scatter(GCOM_times, GCOM_pt['NDVI'].values, color=GCOM_color, s=60, alpha=1,  marker='^', edgecolors='black', linewidths=0.5, zorder = 2)

ax.plot(MCD19_QC_times, MCD19_QC_pt['NDVI'].values, color=MCD19_QC_color, marker = 'o', markersize=8, linewidth=3.5, alpha=1, markeredgecolor='black', markeredgewidth=0.5)

ax.set_ylim(mcd19_ndvi_min - (mcd19_ndvi_range*0.15), mcd19_ndvi_max + (mcd19_ndvi_range*0.05))
ax.yaxis.set_major_formatter(FFmt(lambda y, _: f'{y:.1f}'))
ax.yaxis.set_major_locator(MultipleLocator(0.2))


ax.set_xlim(Photospec_times_filtered.min() - pd.Timedelta(days=10), Photospec_times_filtered.max() + pd.Timedelta(days=10))
ax.xaxis.set_major_locator(MonthLocator(bymonth=[1, 7]))

ax.xaxis.set_major_formatter(DateFormatter('%b\n%Y'))
ax.set_title('')
ax.set_ylabel('NDVI')

# Combine legends using Patch
handles = [
    Patch(color = 'lightblue', label='Snow Cover (PhenoCam)', alpha=0.3),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label='PhotoSpec'),
    plt.Line2D([0], [0], color=MCD19_QC_color, marker = 'o', markersize = 7, linewidth=3, label=f'MCD19 ({MCD19_legend_descr})'),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=MOD13_color, markersize=8, label='MOD13'),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=MCD43_color, markersize=8, label='MCD43'),
    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=MOD09_color, markeredgecolor='black', markersize=10, label='MOD09'),
    # plt.Line2D([0], [0], color=GCOM_color, marker='^', markersize = 9, linewidth=2.5, label='GCOM-C'),
    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=GCOM_color, markeredgecolor='black', markersize=10, label='GCOM-C'),


]
ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.3), fontsize=12, ncol=2)
plt.subplots_adjust(bottom=0.35)

plt.tight_layout()
plt.show()

#%%
fig, ax = plt.subplots(figsize=(7, 3))

for start, end in winter_periods:
    ax.axvspan(start, end, color='lightblue', alpha=0.5)
ax.scatter(Photospec_times_filtered, Photospec_df_filtered['NDVI'], color='black', s=10, alpha=0.7)

ax.scatter(MOD13_times, MOD13_pt['NDVI'].values, color=MOD13_color, s=35, alpha=1, marker='s', edgecolors='black', linewidths=0.5)
ax.scatter(MCD43_times, MCD43_pt['NDVI'].values, color=MCD43_color, s=35, alpha=1, marker='s', edgecolors='black', linewidths=0.5)

ax.scatter(MOD09_times, MOD09_pt['NDVI'].values, color=MOD09_color, s=60, alpha=1,  marker='^', edgecolors='black', linewidths=0.5, zorder = 2)
# ax.plot(GCOM_times, GCOM_pt['NDVI'].values, color=GCOM_color, marker = '^', markersize=9, linewidth=2.5, alpha=1, zorder = 2)
ax.scatter(GCOM_times, GCOM_pt['NDVI'].values, color=GCOM_color, s=60, alpha=1,  marker='^', edgecolors='black', linewidths=0.5, zorder = 2)

ax.plot(MCD19_QC_times, MCD19_QC_pt['NDVI'].values, color=MCD19_QC_color, marker = 'o', markersize=8, linewidth=3.5, alpha=1, markeredgecolor='black', markeredgewidth=0.5)

ax.set_ylim(mcd19_ndvi_min - (mcd19_ndvi_range*0.15), mcd19_ndvi_max + (mcd19_ndvi_range*0.05))
ax.yaxis.set_major_formatter(FFmt(lambda y, _: f'{y:.1f}'))
ax.yaxis.set_major_locator(MultipleLocator(0.2))


ax.set_xlim(Photospec_times_filtered.min() - pd.Timedelta(days=10), Photospec_times_filtered.max() + pd.Timedelta(days=10))
ax.xaxis.set_major_locator(MonthLocator(bymonth=[1, 7]))
ax.xaxis.set_major_formatter(DateFormatter('%b\n%Y'))
ax.set_title('')
ax.set_ylabel('NDVI')

plt.tight_layout()
plt.show()




#%%

### Not as good looking - commenting out 

# ############################################
# ### QC ANALYSIS DELIVERABLE 1 - NDVI time series v2 (two axes)
# ############################################


# # Calculate day of year for MCD19_CloudFree_times and Photospec_times_filtered

# ndvi_arrays = [
#     MCD19_QC_pt['NDVI'].values
# ]
# mcd19_ndvi_min = np.nanmin(np.concatenate(ndvi_arrays))
# mcd19_ndvi_max = np.nanmax(np.concatenate(ndvi_arrays))
# mcd19_ndvi_range = mcd19_ndvi_max - mcd19_ndvi_min


# fig, ax = plt.subplots(figsize=(7, 3))

# for start, end in winter_periods:
#     ax.axvspan(start, end, color='lightblue', alpha=0.5)
# ax.scatter(Photospec_times_filtered, Photospec_df_filtered['NDVI'], color='black', s=10, alpha=0.7)
# ax2 = ax.twinx()

# ax2.scatter(MOD13_times, MOD13_pt['NDVI'].values, color=MOD13_color, s=35, alpha=1, marker='s', edgecolors='black', linewidths=0.5)
# ax2.scatter(MCD43_times, MCD43_pt['NDVI'].values, color=MCD43_color, s=35, alpha=1, marker='s', edgecolors='black', linewidths=0.5)

# ax2.scatter(MOD09_times, MOD09_pt['NDVI'].values, color=MOD09_color, s=60, alpha=1,  marker='^', edgecolors='black', linewidths=0.5, zorder = 2)
# ax2.scatter(GCOM_times, GCOM_pt['NDVI'].values, color=GCOM_color, s=60, alpha=1,  marker='^', edgecolors='black', linewidths=0.5, zorder = 2)

# ax2.plot(MCD19_QC_times, MCD19_QC_pt['NDVI'].values, color=MCD19_QC_color, marker = 'o', markersize=8, linewidth=3.5, alpha=1, markeredgecolor='black', markeredgewidth=0.5)

# ax2.set_ylim(mcd19_ndvi_min - (mcd19_ndvi_range*0.05), mcd19_ndvi_max + (mcd19_ndvi_range*0.05))
# ax.yaxis.set_major_formatter(FFmt(lambda y, _: f'{y:.1f}'))
# ax2.yaxis.set_major_formatter(FFmt(lambda y, _: f'{y:.1f}'))
# ax.yaxis.set_major_locator(MultipleLocator(0.1))
# ax2.yaxis.set_major_locator(MultipleLocator(0.1))

# ax.set_xlim(Photospec_times_filtered.min() - pd.Timedelta(days=10), Photospec_times_filtered.max() + pd.Timedelta(days=10))
# ax.xaxis.set_major_locator(MonthLocator(bymonth=[1, 7]))

# ax.xaxis.set_major_formatter(DateFormatter('%b\n%Y'))
# ax.set_title('')
# ax.set_ylabel('NDVI')

# plt.tight_layout()
# plt.show()


#%%

############################################
### QC ANALYSIS DELIVERABLE - CCI time series (one axis)
############################################

cci_arrays = [
    MCD19_QC_pt['CCI'].values,
    Photospec_df_filtered['CCI']
]
mcd19_cci_min = np.nanmin(np.concatenate(cci_arrays))
mcd19_cci_max = np.nanmax(np.concatenate(cci_arrays))
mcd19_cci_range = mcd19_cci_max - mcd19_cci_min

fig, ax = plt.subplots(figsize=(7, 3))

for start, end in winter_periods:
    ax.axvspan(start, end, color='lightblue', alpha=0.5)
ax.scatter(Photospec_times_filtered, Photospec_df_filtered['CCI'], color='black', s=10, alpha=0.7)

ax.scatter(MOD09_times, MOD09_pt['CCI'].values, color=MOD09_color, s=60, alpha=1,  marker='^', edgecolors='black', linewidths=0.5, zorder = 2)
# ax.plot(GCOM_times, GCOM_pt['CCI'].values, color=GCOM_color, marker = '^', markersize=9, linewidth=2.5, alpha=1, zorder = 2)
ax.scatter(GCOM_times, GCOM_pt['CCI'].values, color=GCOM_color, s=60, alpha=1,  marker='^', edgecolors='black', linewidths=0.5, zorder = 2)
ax.plot(MCD19_QC_times, MCD19_QC_pt['CCI'].values, color=MCD19_QC_color, marker = 'o', markersize=8, linewidth=3.5, alpha=1, markeredgecolor='black', markeredgewidth=0.5)

ax.set_ylim(mcd19_cci_min - (mcd19_cci_range*0.05), mcd19_cci_max + (mcd19_cci_range*0.05))
ax.yaxis.set_major_formatter(FFmt(lambda y, _: f'{y:.1f}'))
ax.yaxis.set_major_formatter(FFmt(lambda y, _: f'{y:.1f}'))
ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax.set_xlim(Photospec_times_filtered.min() - pd.Timedelta(days=10), Photospec_times_filtered.max() + pd.Timedelta(days=10))
ax.xaxis.set_major_locator(MonthLocator(bymonth=[1, 7]))
ax.xaxis.set_major_formatter(DateFormatter('%b\n%Y'))
ax.set_title('')
ax.set_ylabel('CCI')

plt.tight_layout()
plt.show()

# plot_filename = os.path.join(plot_dir, f'MCD19_v_PhotoSpec_timeseries_{out_stats_descr}_CCI_{site_name}.png')
# plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
# plt.close()

#%%

### Actually this one looks much better than on one axis

############################################
### QC ANALYSIS DELIVERABLE 2 - CCI time series v2 (2 axes)
############################################

# Calculate day of year for MCD19_CloudFree_times and Photospec_times_filtered
mcd19_doy = pd.to_datetime(MCD19_QC_times).dayofyear
photospec_doy = Photospec_times_filtered.dt.dayofyear

cci_arrays = [
    MCD19_QC_pt['CCI'].values
]
mcd19_cci_min = np.nanmin(np.concatenate(cci_arrays))
mcd19_cci_max = np.nanmax(np.concatenate(cci_arrays))
mcd19_cci_range = mcd19_cci_max - mcd19_cci_min

fig, ax = plt.subplots(figsize=(7, 3))

for start, end in winter_periods:
    ax.axvspan(start, end, color='lightblue', alpha=0.5)
ax.scatter(Photospec_times_filtered, Photospec_df_filtered['CCI'], color='black', s=10, alpha=0.7)
ax2 = ax.twinx()

ax2.scatter(MOD09_times, MOD09_pt['CCI'].values, color=MOD09_color, s=60, alpha=1,  marker='^', edgecolors='black', linewidths=0.5, zorder = 2)
# ax2.plot(GCOM_times, GCOM_pt['CCI'].values, color=GCOM_color, marker = '^', markersize=9, linewidth=2.5, alpha=1, zorder = 2)
ax2.scatter(GCOM_times, GCOM_pt['CCI'].values, color=GCOM_color, s=60, alpha=1,  marker='^', edgecolors='black', linewidths=0.5, zorder = 2)

ax2.plot(MCD19_QC_times, MCD19_QC_pt['CCI'].values, color=MCD19_QC_color, marker = 'o', markersize=8, linewidth=3.5, alpha=1, markeredgecolor='black', markeredgewidth=0.5)

ax2.set_ylim(mcd19_cci_min - (mcd19_cci_range*0.05), mcd19_cci_max + (mcd19_cci_range*0.05))
ax.yaxis.set_major_formatter(FFmt(lambda y, _: f'{y:.1f}'))
ax2.yaxis.set_major_formatter(FFmt(lambda y, _: f'{y:.1f}'))
ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax2.yaxis.set_major_locator(MultipleLocator(0.1))
ax.set_xlim(Photospec_times_filtered.min() - pd.Timedelta(days=10), Photospec_times_filtered.max() + pd.Timedelta(days=10))
ax.xaxis.set_major_locator(MonthLocator(bymonth=[1, 7]))
ax.xaxis.set_major_formatter(DateFormatter('%b\n%Y'))
ax.set_title('')
ax.set_ylabel('CCI')

plt.tight_layout()
plt.show()

# plot_filename = os.path.join(plot_dir, f'MCD19_v_PhotoSpec_timeseries_{out_stats_descr}_CCI_{site_name}.png')
# plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
# plt.close()

#%%
#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#
#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#

### Calculate fit params

#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#
#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#

# First, Interpolate Photospec CCI to match MCD19_QC_pt.time
photospec_cci_interp = pd.Series(
    np.interp(
        pd.to_datetime(MCD19_QC_pt.time.values).astype(np.int64),
        Photospec_times_filtered.astype(np.int64),
        Photospec_df_filtered['CCI'].values
    ),
    index=MCD19_QC_pt.time.values
)


# First, Interpolate Photospec CCI to match MCD19_QC_pt.time
photospec_ndvi_interp = pd.Series(
    np.interp(
        pd.to_datetime(MCD19_QC_pt.time.values).astype(np.int64),
        Photospec_times_filtered.astype(np.int64),
        Photospec_df_filtered['NDVI'].values
    ),
    index=MCD19_QC_pt.time.values
)


# Set to nan before first and after last valid Photospec time
first_time = Photospec_times_filtered.iloc[0]
last_time = Photospec_times_filtered.iloc[-1]
interp_times = pd.to_datetime(MCD19_QC_pt.time.values)

photospec_cci_interp[(interp_times < first_time) | (interp_times > last_time)] = np.nan
photospec_ndvi_interp[(interp_times < first_time) | (interp_times > last_time)] = np.nan

# Convert interp_times to dayofyear
interp_times_doy = pd.to_datetime(interp_times).dayofyear

# Calculate day of year for MCD19_CloudFree_times and Photospec_times_filtered
mcd19_doy = pd.to_datetime(MCD19_QC_pt.time.values).dayofyear
photospec_doy = Photospec_times_filtered.dt.dayofyear

############################################
### sanity check - plot interpolated Photospec CCI 
############################################
fig, ax = plt.subplots(figsize=(7, 3))

# # Plot shaded region where snow depth exceeds threshold

# Plot Photospec CCI on left y-axis
ax.scatter(photospec_doy, Photospec_df_filtered['CCI'], label='Daily, 1-2pm', color='gray', s=14, alpha=0.5)

# Plot interpolated Photospec CCI on right y-axis
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
valid_mask = (~np.isnan(MCD19_QC_pt['CCI'].values)) & (~np.isnan(photospec_cci_interp.values))
# Compute R^2 for the scatter of interpolated Photospec CCI vs filtered MCD19 CCI
if np.sum(valid_mask) > 1:
    slope, intercept, r_value, p_value, std_err = linregress(
        photospec_cci_interp.values[valid_mask],
        MCD19_QC_pt['CCI'].values[valid_mask]
    )
    # Spearman rank correlation
    spear_r, spear_p = spearmanr(photospec_cci_interp.values[valid_mask], MCD19_QC_pt['CCI'].values[valid_mask])

    # Centered (unbiased) RMSE (CRMSE): removes mean bias
    x = photospec_cci_interp.values[valid_mask]
    y = MCD19_QC_pt['CCI'].values[valid_mask]
    crmse = np.sqrt(np.nanmean(((y - np.nanmean(y)) - (x - np.nanmean(x)))**2))
    bias = np.nanmean(y) - np.nanmean(x)
    cci_fit_stats = [r_value**2, spear_r, crmse, bias]

    print(f"MCD19 vs PhotoSpec CCI Fit parameters (interp): R^2={r_value**2:.3f}, bias={bias:.3f}, CRMSE={crmse:.3f}, Spearman r={spear_r:.3f}, p-value={spear_p:.3e}")
else:
    print("Not enough valid data for regression (interp).")



# Only compare where both are valid (not nan)
valid_mask = (~np.isnan(MCD19_QC_pt['NDVI'].values)) & (~np.isnan(photospec_ndvi_interp.values))
# Compute R^2 for the scatter of interpolated Photospec NDVI vs filtered MCD19 NDVI
if np.sum(valid_mask) > 1:
    slope, intercept, r_value, p_value, std_err = linregress(
        photospec_ndvi_interp.values[valid_mask],
        MCD19_QC_pt['NDVI'].values[valid_mask]
    )
    # Spearman rank correlation
    spear_r, spear_p = spearmanr(photospec_ndvi_interp.values[valid_mask], MCD19_QC_pt['NDVI'].values[valid_mask])

    # Centered (unbiased) RMSE (CRMSE): removes mean bias
    x = photospec_ndvi_interp.values[valid_mask]
    y = MCD19_QC_pt['NDVI'].values[valid_mask]
    crmse = np.sqrt(np.nanmean(((y - np.nanmean(y)) - (x - np.nanmean(x)))**2))
    bias = np.nanmean(y) - np.nanmean(x)
    ndvi_fit_stats = [r_value**2, spear_r, crmse, bias]

    print(f"MCD19 vs PhotoSpec NDVI Fit parameters (interp): R^2={r_value**2:.3f}, bias={bias:.3f}, CRMSE={crmse:.3f}, Spearman r={spear_r:.3f}, p-value={spear_p:.3e}")
else:
    print("Not enough valid data for regression (interp).")



######################################################################
### For comparison, check PhotoSpec vs. MOD09 fit params
######################################################################

MOD09_interp = MOD09_pt.sel(time=slice(photospec_cci_interp.index[0], Photospec_times_filtered.max()+pd.Timedelta(days=16)))

# Check that the interpolated MOD09 series is aligned with photospec_cci_interp.
mod09_interp_times = pd.DatetimeIndex(pd.to_datetime(MOD09_interp.time.values))
photospec_cci_interp_times = pd.DatetimeIndex(pd.to_datetime(photospec_cci_interp.index))
if not mod09_interp_times.equals(photospec_cci_interp_times):
    raise ValueError("MOD09_interp and photospec_cci_interp timestamps do not match")
print(f"MOD09_interp timestamps match photospec_cci_interp ({len(MOD09_interp.time)} timestamps)")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
### sanity check - plot interpolated Photospec CCI vs MOD09 CCI
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mod09_doy = pd.to_datetime(MOD09_interp.time.values).dayofyear

fig, ax = plt.subplots(figsize=(7, 3))

# # Plot shaded region where snow depth exceeds threshold

# Plot Photospec CCI on left y-axis
ax.scatter(photospec_doy, Photospec_df_filtered['CCI'], label='Daily, 1-2pm', color='gray', s=14, alpha=0.5)

# Plot interpolated Photospec CCI on right y-axis
ax.scatter(interp_times_doy, photospec_cci_interp.values, label='16-day', color='black', s=28, alpha=1, marker='^')

ax.scatter(mod09_doy, MOD09_interp['CCI'].values, label='MOD09 16-day', color='red', s=28, alpha=1, marker='s')

# Adjust limits and legend
ax.set_ylabel('Photospec CCI')
ax.set_xticks(month_starts)
ax.set_xticklabels(month_labels)

ax.set_title(site_name)
ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()


# Only compare where both are valid (not nan)
valid_mask = (~np.isnan(MOD09_interp['CCI'].values)) & (~np.isnan(photospec_cci_interp.values))
# Compute R^2 for the scatter of interpolated Photospec NDVI vs filtered MCD19 NDVI
if np.sum(valid_mask) > 1:
    slope, intercept, r_value, p_value, std_err = linregress(
        photospec_cci_interp.values[valid_mask],
        MOD09_interp['CCI'].values[valid_mask]
    )
    # Spearman rank correlation
    spear_r, spear_p = spearmanr(photospec_cci_interp.values[valid_mask], MOD09_interp['CCI'].values[valid_mask])

    # Centered (unbiased) RMSE (CRMSE): removes mean bias
    x = photospec_cci_interp.values[valid_mask]
    y = MOD09_interp['CCI'].values[valid_mask]
    crmse = np.sqrt(np.nanmean(((y - np.nanmean(y)) - (x - np.nanmean(x)))**2))
    bias = np.nanmean(y) - np.nanmean(x)
    mod09_cci_fit_stats = [r_value**2, spear_r, crmse, bias]

    print(f"MOD09 vs PhotoSpec CCI Fit parameters (interp): R^2={r_value**2:.3f}, bias={bias:.3f}, CRMSE={crmse:.3f}, Spearman r={spear_r:.3f}, p-value={spear_p:.3e}")
else:
    print("Not enough valid data for regression (interp).")




# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
### sanity check - plot interpolated Photospec NDVI vs MOD09 NDVI
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

fig, ax = plt.subplots(figsize=(7, 3))

# # Plot shaded region where snow depth exceeds threshold

# Plot Photospec NDVI on left y-axis
ax.scatter(photospec_doy, Photospec_df_filtered['NDVI'], label='Daily, 1-2pm', color='gray', s=14, alpha=0.5)

# Plot interpolated Photospec NDVI on right y-axis
ax.scatter(interp_times_doy, photospec_ndvi_interp.values, label='16-day', color='black', s=28, alpha=1, marker='^')

ax.scatter(mod09_doy, MOD09_interp['NDVI'].values, label='MOD09 16-day', color='red', s=28, alpha=1, marker='s')

# Adjust limits and legend
ax.set_ylabel('Photospec NDVI')
ax.set_xticks(month_starts)
ax.set_xticklabels(month_labels)

ax.set_title(site_name)
ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()


# Only compare where both are valid (not nan)
valid_mask = (~np.isnan(MOD09_interp['NDVI'].values)) & (~np.isnan(photospec_ndvi_interp.values))
# Compute R^2 for the scatter of interpolated Photospec NDVI vs filtered MCD19 NDVI
if np.sum(valid_mask) > 1:
    slope, intercept, r_value, p_value, std_err = linregress(
        photospec_ndvi_interp.values[valid_mask],
        MOD09_interp['NDVI'].values[valid_mask]
    )
    # Spearman rank correlation
    spear_r, spear_p = spearmanr(photospec_ndvi_interp.values[valid_mask], MOD09_interp['NDVI'].values[valid_mask])

    # Centered (unbiased) RMSE (CRMSE): removes mean bias
    x = photospec_ndvi_interp.values[valid_mask]
    y = MOD09_interp['NDVI'].values[valid_mask]
    crmse = np.sqrt(np.nanmean(((y - np.nanmean(y)) - (x - np.nanmean(x)))**2))
    bias = np.nanmean(y) - np.nanmean(x)
    ndvi_fit_stats = [r_value**2, spear_r, crmse, bias]

    print(f"MOD09 vs PhotoSpec NDVI Fit parameters (interp): R^2={r_value**2:.3f}, bias={bias:.3f}, CRMSE={crmse:.3f}, Spearman r={spear_r:.3f}, p-value={spear_p:.3e}")
else:
    print("Not enough valid data for regression (interp).")




######################################################################
### For comparison, check PhotoSpec vs. MOD13 fit params
######################################################################

MOD13_interp = MOD13_pt.sel(time=slice(photospec_cci_interp.index[0], Photospec_times_filtered.max()+pd.Timedelta(days=16)))

# Check that the interpolated MOD13 series is aligned with photospec_cci_interp.
MOD13_interp_times = pd.DatetimeIndex(pd.to_datetime(MOD13_interp.time.values))
if not MOD13_interp_times.equals(photospec_cci_interp_times):
    raise ValueError("MOD13_interp and photospec_cci_interp timestamps do not match")
print(f"MOD13_interp timestamps match photospec_cci_interp ({len(MOD13_interp.time)} timestamps)")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
### sanity check - plot interpolated Photospec NDVI vs MOD13 NDVI
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mod13_doy = pd.to_datetime(MOD13_interp.time.values).dayofyear

fig, ax = plt.subplots(figsize=(7, 3))

# # Plot shaded region where snow depth exceeds threshold

# Plot Photospec NDVI on left y-axis
ax.scatter(photospec_doy, Photospec_df_filtered['NDVI'], label='Daily, 1-2pm', color='gray', s=14, alpha=0.5)

# Plot interpolated Photospec NDVI on right y-axis
ax.scatter(interp_times_doy, photospec_ndvi_interp.values, label='16-day', color='black', s=28, alpha=1, marker='^')

ax.scatter(mod13_doy, MOD13_interp['NDVI'].values, label='MOD13 16-day', color='red', s=28, alpha=1, marker='s')

# Adjust limits and legend
ax.set_ylabel('Photospec NDVI')
ax.set_xticks(month_starts)
ax.set_xticklabels(month_labels)

ax.set_title(site_name)
ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()


# Only compare where both are valid (not nan)
valid_mask = (~np.isnan(MOD13_interp['NDVI'].values)) & (~np.isnan(photospec_ndvi_interp.values))
# Compute R^2 for the scatter of interpolated Photospec NDVI vs filtered MCD19 NDVI
if np.sum(valid_mask) > 1:
    slope, intercept, r_value, p_value, std_err = linregress(
        photospec_ndvi_interp.values[valid_mask],
        MOD13_interp['NDVI'].values[valid_mask]
    )
    # Spearman rank correlation
    spear_r, spear_p = spearmanr(photospec_ndvi_interp.values[valid_mask], MOD13_interp['NDVI'].values[valid_mask])

    # Centered (unbiased) RMSE (CRMSE): removes mean bias
    x = photospec_ndvi_interp.values[valid_mask]
    y = MOD13_interp['NDVI'].values[valid_mask]
    crmse = np.sqrt(np.nanmean(((y - np.nanmean(y)) - (x - np.nanmean(x)))**2))
    bias = np.nanmean(y) - np.nanmean(x)
    ndvi_fit_stats = [r_value**2, spear_r, crmse, bias]

    print(f"MOD13 vs PhotoSpec NDVI Fit parameters (interp): R^2={r_value**2:.3f}, bias={bias:.3f}, CRMSE={crmse:.3f}, Spearman r={spear_r:.3f}, p-value={spear_p:.3e}")
else:
    print("Not enough valid data for regression (interp).")




######################################################################
### For comparison, check PhotoSpec vs. MOD13 fit params
######################################################################


# First, Interpolate Photospec CCI to match MCD19_QC_pt.time
photospec_cci_interp_FOR_GCOM_COMPARE = pd.Series(
    np.interp(
        pd.to_datetime(GCOM_pt.time.values).astype(np.int64),
        Photospec_times_filtered.astype(np.int64),
        Photospec_df_filtered['CCI'].values
    ),
    index=GCOM_pt.time.values
)


# First, Interpolate Photospec CCI to match MCD19_QC_pt.time
photospec_ndvi_interp_FOR_GCOM_COMPARE = pd.Series(
    np.interp(
        pd.to_datetime(GCOM_pt.time.values).astype(np.int64),
        Photospec_times_filtered.astype(np.int64),
        Photospec_df_filtered['NDVI'].values
    ),
    index=GCOM_pt.time.values
)


# Set to nan before first and after last valid Photospec time
first_time = Photospec_times_filtered.iloc[0]
last_time = Photospec_times_filtered.iloc[-1]
interp_times = pd.to_datetime(GCOM_pt.time.values)

photospec_cci_interp_FOR_GCOM_COMPARE[(interp_times < first_time) | (interp_times > last_time)] = np.nan
photospec_ndvi_interp_FOR_GCOM_COMPARE[(interp_times < first_time) | (interp_times > last_time)] = np.nan

# if not GCOM_pt.time.equals(photospec_cci_interp_times_FOR_GCOM_COMPARE):
#     raise ValueError("GCOM_interp and photospec_cci_interp timestamps do not match")
# print(f"GCOM_interp timestamps match photospec_cci_interp ({len(GCOM_interp.time)} timestamps)")



# Only compare where both are valid (not nan)
valid_mask = (~np.isnan(GCOM_pt['CCI'].values)) & (~np.isnan(photospec_cci_interp_FOR_GCOM_COMPARE.values))
# Compute R^2 for the scatter of interpolated Photospec CCI vs filtered MCD19 CCI
if np.sum(valid_mask) > 1:
    slope, intercept, r_value, p_value, std_err = linregress(
        photospec_cci_interp_FOR_GCOM_COMPARE.values[valid_mask],
        GCOM_pt['CCI'].values[valid_mask]
    )
    # Spearman rank correlation
    spear_r, spear_p = spearmanr(photospec_cci_interp_FOR_GCOM_COMPARE.values[valid_mask], GCOM_pt['CCI'].values[valid_mask])

    # Centered (unbiased) RMSE (CRMSE): removes mean bias
    x = photospec_cci_interp_FOR_GCOM_COMPARE.values[valid_mask]
    y = GCOM_pt['CCI'].values[valid_mask]
    crmse = np.sqrt(np.nanmean(((y - np.nanmean(y)) - (x - np.nanmean(x)))**2))
    bias = np.nanmean(y) - np.nanmean(x)
    cci_fit_stats = [r_value**2, spear_r, crmse, bias]

    print(f"GCOM vs PhotoSpec CCI Fit parameters (interp): R^2={r_value**2:.3f}, bias={bias:.3f}, CRMSE={crmse:.3f}, Spearman r={spear_r:.3f}, p-value={spear_p:.3e}")
else:
    print("Not enough valid data for regression (interp).")


# Only compare where both are valid (not nan)
valid_mask = (~np.isnan(GCOM_pt['NDVI'].values)) & (~np.isnan(photospec_ndvi_interp_FOR_GCOM_COMPARE.values))
# Compute R^2 for the scatter of interpolated Photospec NDVI vs filtered MCD19 NDVI
if np.sum(valid_mask) > 1:
    slope, intercept, r_value, p_value, std_err = linregress(
        photospec_ndvi_interp_FOR_GCOM_COMPARE.values[valid_mask],
        GCOM_pt['NDVI'].values[valid_mask]
    )
    # Spearman rank correlation
    spear_r, spear_p = spearmanr(photospec_ndvi_interp_FOR_GCOM_COMPARE.values[valid_mask], GCOM_pt['NDVI'].values[valid_mask])

    # Centered (unbiased) RMSE (CRMSE): removes mean bias
    x = photospec_ndvi_interp_FOR_GCOM_COMPARE.values[valid_mask]
    y = GCOM_pt['NDVI'].values[valid_mask]
    crmse = np.sqrt(np.nanmean(((y - np.nanmean(y)) - (x - np.nanmean(x)))**2))
    bias = np.nanmean(y) - np.nanmean(x)
    ndvi_fit_stats = [r_value**2, spear_r, crmse, bias]

    print(f"GCOM vs PhotoSpec NDVI Fit parameters (interp): R^2={r_value**2:.3f}, bias={bias:.3f}, CRMSE={crmse:.3f}, Spearman r={spear_r:.3f}, p-value={spear_p:.3e}")
else:
    print("Not enough valid data for regression (interp).")




######################################################################
### For comparison, check PhotoSpec vs. MCD43 fit params
######################################################################

MCD43_interp = MCD43_pt.interp(time=photospec_ndvi_interp.index)

# Check that the interpolated MCD43 series is aligned with photospec_cci_interp.
MCD43_interp_times = pd.DatetimeIndex(pd.to_datetime(MCD43_interp.time.values))
if not MCD43_interp_times.equals(photospec_cci_interp_times):
    raise ValueError("MCD43_interp and photospec_cci_interp timestamps do not match")
print(f"MCD43_interp timestamps match photospec_cci_interp ({len(MCD43_interp.time)} timestamps)")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
### sanity check - plot interpolated Photospec NDVI vs MCD43 NDVI
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mcd43_doy = pd.to_datetime(MCD43_interp.time.values).dayofyear

fig, ax = plt.subplots(figsize=(7, 3))

# # Plot shaded region where snow depth exceeds threshold

# Plot Photospec NDVI on left y-axis
ax.scatter(photospec_doy, Photospec_df_filtered['NDVI'], label='Daily, 1-2pm', color='gray', s=14, alpha=0.5)

# Plot interpolated Photospec NDVI on right y-axis
ax.scatter(interp_times_doy, photospec_ndvi_interp.values, label='16-day', color='black', s=28, alpha=1, marker='^')

ax.scatter(mcd43_doy, MCD43_interp['NDVI'].values, label='MCD43 16-day', color='red', s=28, alpha=1, marker='s')

# Adjust limits and legend
ax.set_ylabel('Photospec NDVI')
ax.set_xticks(month_starts)
ax.set_xticklabels(month_labels)

ax.set_title(site_name)
ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()


# Only compare where both are valid (not nan)
valid_mask = (~np.isnan(MCD43_interp['NDVI'].values)) & (~np.isnan(photospec_ndvi_interp.values))
# Compute R^2 for the scatter of interpolated Photospec NDVI vs filtered MCD19 NDVI
if np.sum(valid_mask) > 1:
    slope, intercept, r_value, p_value, std_err = linregress(
        photospec_ndvi_interp.values[valid_mask],
        MCD43_interp['NDVI'].values[valid_mask]
    )
    # Spearman rank correlation
    spear_r, spear_p = spearmanr(photospec_ndvi_interp.values[valid_mask], MCD43_interp['NDVI'].values[valid_mask])

    # Centered (unbiased) RMSE (CRMSE): removes mean bias
    x = photospec_ndvi_interp.values[valid_mask]
    y = MCD43_interp['NDVI'].values[valid_mask]
    crmse = np.sqrt(np.nanmean(((y - np.nanmean(y)) - (x - np.nanmean(x)))**2))
    bias = np.nanmean(y) - np.nanmean(x)
    ndvi_fit_stats = [r_value**2, spear_r, crmse, bias]

    print(f"MCD43 vs PhotoSpec NDVI Fit parameters (interp): R^2={r_value**2:.3f}, bias={bias:.3f}, CRMSE={crmse:.3f}, Spearman r={spear_r:.3f}, p-value={spear_p:.3e}")
else:
    print("Not enough valid data for regression (interp).")
