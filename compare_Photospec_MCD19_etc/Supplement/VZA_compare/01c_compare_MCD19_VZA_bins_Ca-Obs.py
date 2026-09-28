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
site_name = 'Ca-Obs'
MODIS_tile = 'h11v03'


QC_descr = "CloudFree_LowAOD_ClearAdj"
VZA1_descr = "VZA0-15"
VZA2_descr = "VZA15-30"
VZA3_descr = "VZA30-45"
VZA4_descr = "VZA45-60"

VZA1_legend_descr = 'VZA 0-15°'
VZA2_legend_descr = 'VZA 15-30°'
VZA3_legend_descr = 'VZA 30-45°'
VZA4_legend_descr = 'VZA 45-60°'

dat_pt_basedir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/data/QC_PhotoSpec_sites_backup/'
dat_pt_basedir_VZA1 = os.path.join(dat_pt_basedir, f'CV-MVC/pre-fill/VZA0-15/QCfilt_{QC_descr}')
dat_pt_basedir_VZA2 = os.path.join(dat_pt_basedir, f'CV-MVC/pre-fill/VZA15-30/QCfilt_{QC_descr}')
dat_pt_basedir_VZA3 = os.path.join(dat_pt_basedir, f'CV-MVC/pre-fill/VZA30-45/QCfilt_{QC_descr}')
dat_pt_basedir_VZA4 = os.path.join(dat_pt_basedir, f'CV-MVC/pre-fill/VZA45-60/QCfilt_{QC_descr}')

MCD19_VZA1_file = os.path.join(dat_pt_basedir_VZA1, f'MCD19_QCfilt_{QC_descr}_{site_name}.nc')
MCD19_VZA2_file = os.path.join(dat_pt_basedir_VZA2, f'MCD19_QCfilt_{QC_descr}_{site_name}.nc')
MCD19_VZA3_file = os.path.join(dat_pt_basedir_VZA3, f'MCD19_QCfilt_{QC_descr}_{site_name}.nc')
MCD19_VZA4_file = os.path.join(dat_pt_basedir_VZA4, f'MCD19_QCfilt_{QC_descr}_{site_name}.nc')

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



MCD19_VZA1_pt = xr.open_dataset(MCD19_VZA1_file)
MCD19_VZA2_pt = xr.open_dataset(MCD19_VZA2_file)
MCD19_VZA3_pt = xr.open_dataset(MCD19_VZA3_file)
MCD19_VZA4_pt = xr.open_dataset(MCD19_VZA4_file)

#### SPECIFIC TO THIS VERSION HERE - re-order if necessary
all_times = np.union1d(np.union1d(np.union1d(MCD19_VZA1_pt.time.values, MCD19_VZA2_pt.time.values), MCD19_VZA3_pt.time.values), MCD19_VZA4_pt.time.values)
MCD19_VZA1_pt = MCD19_VZA1_pt.reindex(time=all_times)
MCD19_VZA2_pt = MCD19_VZA2_pt.reindex(time=all_times)
MCD19_VZA3_pt = MCD19_VZA3_pt.reindex(time=all_times)
MCD19_VZA4_pt = MCD19_VZA4_pt.reindex(time=all_times)


#%%

Photospec_df = pd.read_csv(Photospec_file)
print("Photospec_df columns:", Photospec_df.columns.tolist())

# Print the time range in the 'Time' column of Photospec_df
print(f"Photospec time range: {Photospec_df['Time'].iloc[0]} to {Photospec_df['Time'].iloc[-1]}")

# Remove any data from Photospec_df where NDVI is below 0.4
Photospec_df = Photospec_df[Photospec_df['NDVI'] >= 0.4].reset_index(drop=True)
# Convert Photospec timestamps from "M/D/YY HH:MM" format to datetime64
Photospec_times = pd.to_datetime(Photospec_df['Time'], format='%m/%d/%y %H:%M')



MCD19_VZA1_times = np.array([np.datetime64(t) for t in MCD19_VZA1_pt['time'].values])
MCD19_VZA2_times = np.array([np.datetime64(t) for t in MCD19_VZA2_pt['time'].values])
MCD19_VZA3_times = np.array([np.datetime64(t) for t in MCD19_VZA3_pt['time'].values])
MCD19_VZA4_times = np.array([np.datetime64(t) for t in MCD19_VZA4_pt['time'].values])
# GCOM_times = np.array([np.datetime64(t) for t in GCOM_pt['time'].values])   


# Filter Photospec_df for times between 13:00 and 14:00
mask = (Photospec_times.dt.hour >= 13) & (Photospec_times.dt.hour < 14)
Photospec_df_filtered = Photospec_df[mask].reset_index(drop=True)
Photospec_times_filtered = Photospec_times[mask].reset_index(drop=True)

# Set x-axis to month abbreviations at the start of each month
months = np.arange(1, 13)
month_starts = [pd.Timestamp(f'{composite_years[0]}-{m:02d}-01').dayofyear for m in months]
month_labels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']

# Calculate day of year for MCD19_CloudFree_times and Photospec_times_filtered
MCD19_VZA1_times = np.array([np.datetime64(t) for t in MCD19_VZA1_pt['time'].values])
MCD19_VZA2_times = np.array([np.datetime64(t) for t in MCD19_VZA2_pt['time'].values])
MCD19_VZA3_times = np.array([np.datetime64(t) for t in MCD19_VZA3_pt['time'].values])
MCD19_VZA4_times = np.array([np.datetime64(t) for t in MCD19_VZA4_pt['time'].values])

mcd19_doy = pd.to_datetime(MCD19_VZA1_times).dayofyear
photospec_doy = Photospec_times_filtered.dt.dayofyear


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
MCD19_VZA1_color = '#F5D327'
MCD19_VZA2_color = '#DB830B' 
MCD19_VZA3_color = '#BF2665'
MCD19_VZA4_color = '#9F48B5'

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


##############################################
## COMPARE NDVI
##############################################


ndvi_arrays = [
    MCD19_VZA1_pt['NDVI'].values,
    MCD19_VZA2_pt['NDVI'].values,
    MCD19_VZA3_pt['NDVI'].values,
    MCD19_VZA4_pt['NDVI'].values,
]
mcd19_ndvi_min = np.nanmin(np.concatenate(ndvi_arrays))
mcd19_ndvi_max = np.nanmax(np.concatenate(ndvi_arrays))
mcd19_ndvi_range = mcd19_ndvi_max - mcd19_ndvi_min


xmin = dt(2018, 6, 1)
xmax = dt(2022, 8, 30)
# Plot time series of NDVI from MCD19 options
fig, ax = plt.subplots(figsize=(7, 5))


for start, end in winter_periods:
    ax.axvspan(start, end, color='lightblue', alpha=0.5)


ax.scatter(Photospec_times_filtered, Photospec_df_filtered['NDVI'], color='black', s=12, alpha=0.6)
ax2 = ax.twinx()
ax.set_facecolor('#D9D9D9')
ax2.set_facecolor('#D9D9D9')


ax2.plot(MCD19_VZA1_times, MCD19_VZA1_pt['NDVI'].values, color=MCD19_VZA1_color, marker = 'o', markersize=4, linewidth=2, alpha=1, zorder=2, markeredgecolor='white', markeredgewidth=0.5)
ax2.plot(MCD19_VZA2_times, MCD19_VZA2_pt['NDVI'].values, color=MCD19_VZA2_color, marker = 'o', markersize=4, linewidth=2, alpha=1, zorder=2, markeredgecolor='white', markeredgewidth=0.5)
ax2.plot(MCD19_VZA3_times, MCD19_VZA3_pt['NDVI'].values, color=MCD19_VZA3_color, marker = 'o', markersize=4, linewidth=2, alpha=1, zorder=2, markeredgecolor='white', markeredgewidth=0.5)
ax2.plot(MCD19_VZA4_times, MCD19_VZA4_pt['NDVI'].values, color=MCD19_VZA4_color, marker = 'o', markersize=4, linewidth=2, alpha=1, zorder=2, markeredgecolor='white', markeredgewidth=0.5)


ax2.set_ylim(mcd19_ndvi_min - (mcd19_ndvi_range*0.05), mcd19_ndvi_max + (mcd19_ndvi_range*0.05))

ax.set_ylabel('NDVI')
ax.set_xlim(xmin, xmax)
ax.yaxis.set_major_formatter(FFmt(lambda y, _: f'{y:.1f}'))
ax2.yaxis.set_major_formatter(FFmt(lambda y, _: f'{y:.1f}'))

ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax2.yaxis.set_major_locator(MultipleLocator(0.3))

ax.xaxis.set_major_locator(MonthLocator(bymonth=[1]))
ax.xaxis.set_major_formatter(DateFormatter('%b\n%Y'))
ax.set_title('')



# Combine legends using Patch
handles = [
    Patch(color = 'lightblue', label=f'Estimated snow cover timing', alpha=0.5),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label='PhotoSpec'),
    plt.Line2D([0], [0], color=MCD19_VZA1_color, marker='o', markersize = 4, linewidth=2, label=f'MCD19 ({VZA1_legend_descr})'),
    plt.Line2D([0], [0], color=MCD19_VZA2_color, marker='o', markersize = 4, linewidth=2, label=f'MCD19 ({VZA2_legend_descr})'),
    plt.Line2D([0], [0], color=MCD19_VZA3_color, marker='o', markersize = 4, linewidth=2, label=f'MCD19 ({VZA3_legend_descr})'),
    plt.Line2D([0], [0], color=MCD19_VZA4_color, marker='o', markersize = 4, linewidth=2, label=f'MCD19 ({VZA4_legend_descr})')
]
ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.2), fontsize=10, ncol=1)
plt.subplots_adjust(bottom=0.35)

# ax.xaxis.set_major_formatter(DateFormatter('%Y'))
# fig.autofmt_xdate(rotation=45)
plt.tight_layout()
plt.show()


#%%
fig, ax = plt.subplots(figsize=(7, 3.5))


for start, end in winter_periods:
    ax.axvspan(start, end, color='lightblue', alpha=0.5)


ax.scatter(Photospec_times_filtered, Photospec_df_filtered['NDVI'], color='black', s=12, alpha=0.6)
ax2 = ax.twinx()
ax.set_facecolor('#D9D9D9')
ax2.set_facecolor('#D9D9D9')


ax2.plot(MCD19_VZA1_times, MCD19_VZA1_pt['NDVI'].values, color=MCD19_VZA1_color, marker = 'o', markersize=4, linewidth=2, alpha=1, zorder=2, markeredgecolor='white', markeredgewidth=0.5)
ax2.plot(MCD19_VZA2_times, MCD19_VZA2_pt['NDVI'].values, color=MCD19_VZA2_color, marker = 'o', markersize=4, linewidth=2, alpha=1, zorder=2, markeredgecolor='white', markeredgewidth=0.5)
ax2.plot(MCD19_VZA3_times, MCD19_VZA3_pt['NDVI'].values, color=MCD19_VZA3_color, marker = 'o', markersize=4, linewidth=2, alpha=1, zorder=2, markeredgecolor='white', markeredgewidth=0.5)
ax2.plot(MCD19_VZA4_times, MCD19_VZA4_pt['NDVI'].values, color=MCD19_VZA4_color, marker = 'o', markersize=4, linewidth=2, alpha=1, zorder=2, markeredgecolor='white', markeredgewidth=0.5)


ax2.set_ylim(mcd19_ndvi_min - (mcd19_ndvi_range*0.05), mcd19_ndvi_max + (mcd19_ndvi_range*0.05))

ax.set_ylabel('NDVI')
ax.set_xlim(xmin, xmax)
ax.yaxis.set_major_formatter(FFmt(lambda y, _: f'{y:.1f}'))
ax2.yaxis.set_major_formatter(FFmt(lambda y, _: f'{y:.1f}'))

ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax2.yaxis.set_major_locator(MultipleLocator(0.3))

ax.xaxis.set_major_locator(MonthLocator(bymonth=[1]))
ax.xaxis.set_major_formatter(DateFormatter('%b\n%Y'))
ax.set_title('')


# ax.xaxis.set_major_formatter(DateFormatter('%Y'))
# fig.autofmt_xdate(rotation=45)
plt.tight_layout()
plt.show()


#%%



##############################################
## COMPARE CCI
##############################################
cci_arrays = [
    MCD19_VZA1_pt['CCI'].values,
    MCD19_VZA2_pt['CCI'].values,
    MCD19_VZA3_pt['CCI'].values,
    MCD19_VZA4_pt['CCI'].values,
]
mcd19_cci_min = np.nanmin(np.concatenate(cci_arrays))
mcd19_cci_max = np.nanmax(np.concatenate(cci_arrays))
mcd19_cci_range = mcd19_cci_max - mcd19_cci_min


# Plot time series of CCI from MCD19_CloudFree_pt and MCD19_basicfill_pt
fig, ax = plt.subplots(figsize=(7, 3.5))

for start, end in winter_periods:
    ax.axvspan(start, end, color='lightblue', alpha=0.5)


ax.scatter(Photospec_times_filtered, Photospec_df_filtered['CCI'], color='black', s=12, alpha=0.8)
ax2 = ax.twinx()
ax.set_facecolor('#D9D9D9')
ax2.set_facecolor('#D9D9D9')

ax2.plot(MCD19_VZA1_times, MCD19_VZA1_pt['CCI'].values, color=MCD19_VZA1_color, marker = 'o', markersize=4, linewidth=2, alpha=1, zorder=2, markeredgecolor='white', markeredgewidth=0.5)
ax2.plot(MCD19_VZA2_times, MCD19_VZA2_pt['CCI'].values, color=MCD19_VZA2_color, marker = 'o', markersize=4, linewidth=2, alpha=1, zorder=2, markeredgecolor='white', markeredgewidth=0.5)
ax2.plot(MCD19_VZA3_times, MCD19_VZA3_pt['CCI'].values, color=MCD19_VZA3_color, marker = 'o', markersize=4, linewidth=2, alpha=1, zorder=2, markeredgecolor='white', markeredgewidth=0.5)
ax2.plot(MCD19_VZA4_times, MCD19_VZA4_pt['CCI'].values, color=MCD19_VZA4_color, marker = 'o', markersize=4, linewidth=2, alpha=1, zorder=2, markeredgecolor='white', markeredgewidth=0.5)


ax2.set_ylim(mcd19_cci_min - (mcd19_cci_range*0.05), mcd19_cci_max + (mcd19_cci_range*0.05))

ax.set_ylabel('CCI')
ax.set_xlim(xmin, xmax)
ax.set_ylim(-0.2, 0.25)
ax2.set_ylim(-0.2, 0.3)

ax.yaxis.set_major_formatter(FFmt(lambda y, _: f'{y:.1f}'))
ax2.yaxis.set_major_formatter(FFmt(lambda y, _: f'{y:.1f}'))
ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax2.yaxis.set_major_locator(MultipleLocator(0.1))
ax.xaxis.set_major_locator(MonthLocator(bymonth=[1]))
ax.xaxis.set_major_formatter(DateFormatter('%b\n%Y'))
ax.set_title('')


# ax.xaxis.set_major_formatter(DateFormatter('%Y'))
# fig.autofmt_xdate(rotation=45)
plt.tight_layout()
plt.show()



# %%
