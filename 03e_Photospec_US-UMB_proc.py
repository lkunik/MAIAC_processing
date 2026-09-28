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
site_name = 'US-UMB'


Photospec_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/data/PhotoSpec/US-UMB/'
Photospec_2018_file = os.path.join(Photospec_dir, f'PhotoSpecM1_2018_v20260903.nc')
Photospec_2019_file = os.path.join(Photospec_dir, f'PhotoSpecM1_2019_v20260903.nc')




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



#%%

Photospec_xr_2018 = xr.open_dataset(Photospec_2018_file)
Photospec_xr_2019 = xr.open_dataset(Photospec_2019_file)


print(f'{Photospec_xr_2018.HOD.values}')
# Quick diagnostic: CCI at HOD=13, by DOY, for each year
for yr, ds in [('2018', Photospec_xr_2018), ('2019', Photospec_xr_2019)]:
    for hod in range(0, 16):
        cci_hod = ds['CCI'].sel(HOD=hod)
        print(f'HOD index = {hod}')
        n_valid = int(np.isfinite(cci_hod.values).sum())
        n_total = cci_hod.size

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(cci_hod['DOY'].values, cci_hod.values, s=10, color='black', alpha=0.7)
        ax.set_xlabel('DOY')
        ax.set_ylabel('CCI')
        ax.set_title(f'{yr} CCI at HOD={hod*1.5} (valid: {n_valid}/{n_total})')
        plt.tight_layout()
        plt.show()




# Format Photospec data into a single DataFrame with datetime index (like other PS datasets)
Photospec_df_2018 = Photospec_xr_2018.sel(HOD=9).to_dataframe().reset_index()

Photospec_df_2018['Time'] = (
    pd.Timestamp('2018-01-01')
    + pd.to_timedelta(Photospec_df_2018['DOY'].astype(int), unit='D')
    + pd.Timedelta(hours=13, minutes=30)
)


Photospec_df_2019 = Photospec_xr_2019.sel(HOD=9).to_dataframe().reset_index()

Photospec_df_2019['Time'] = (
    pd.Timestamp('2019-01-01')
    + pd.to_timedelta(Photospec_df_2019['DOY'].astype(int), unit='D')
    + pd.Timedelta(hours=13, minutes=30)
)

Photospec_df = pd.concat([Photospec_df_2018, Photospec_df_2019], ignore_index=True)

# Remove any data from Photospec_df where NDVI is below 0.6
Photospec_df = Photospec_df[Photospec_df['NDVI'] >= 0.6].reset_index(drop=True)
# Convert Photospec timestamps from "M/D/YY HH:MM" format to datetime64
Photospec_times = pd.to_datetime(Photospec_df['Time'], format='%m/%d/%y %H:%M')




Photospec_xr_2018_HOD13 = Photospec_xr_2018.sel(HOD=13)
Photospec_xr_2019_HOD13 = Photospec_xr_2019.sel(HOD=13)

# Convert each year's DOY values to datetime64 before combining the datasets.
Photospec_xr_2018_HOD13 = Photospec_xr_2018_HOD13.assign_coords(
    time=(
        'DOY',
        np.datetime64('2018-01-01')
        + (Photospec_xr_2018_HOD13['DOY'].values.astype(int)).astype('timedelta64[D]'),
    )
)
Photospec_xr_2019_HOD13 = Photospec_xr_2019_HOD13.assign_coords(
    time=(
        'DOY',
        np.datetime64('2019-01-01')
        + (Photospec_xr_2019_HOD13['DOY'].values.astype(int)).astype('timedelta64[D]'),
    )
)
Photospec_xr_HOD13 = xr.concat(
    [Photospec_xr_2018_HOD13, Photospec_xr_2019_HOD13], dim='DOY'
)
Photospec_xr_HOD13 = Photospec_xr_HOD13.swap_dims({'DOY': 'time'}).sortby('time')

Photospec_times_filtered = Photospec_xr_HOD13['time'].values


fig, ax = plt.subplots(figsize=(7, 4.5))

ax.scatter(Photospec_times_filtered, Photospec_xr_HOD13['CCI'].values, color='black', s=10, alpha=0.7)

ax.set_xlim(Photospec_times_filtered.min() - pd.Timedelta(days=10), Photospec_times_filtered.max() + pd.Timedelta(days=10))
ax.xaxis.set_major_locator(MonthLocator(bymonth=[7, 10], interval=1))
ax.xaxis.set_major_locator(MonthLocator(bymonth=[1, 4, 7, 10]))
ax.xaxis.set_major_formatter(DateFormatter('%b\n%Y'))
ax.set_title('')
ax.set_ylabel('CCI')
plt.tight_layout()
plt.show()












# Select MCD19 data within the time range of Photospec data
MCD19_QC1_pt = MCD19_QC1_pt.sel(time=slice(Photospec_times.iloc[0], Photospec_times.iloc[-1]))
MCD19_QC2_pt = MCD19_QC2_pt.sel(time=slice(Photospec_times.iloc[0], Photospec_times.iloc[-1]))
MCD19_QC3_pt = MCD19_QC3_pt.sel(time=slice(Photospec_times.iloc[0], Photospec_times.iloc[-1]))



MCD19_QC1_times = np.array([np.datetime64(t) for t in MCD19_QC1_pt['time'].values])
MCD19_QC2_times = np.array([np.datetime64(t) for t in MCD19_QC2_pt['time'].values])
MCD19_QC3_times = np.array([np.datetime64(t) for t in MCD19_QC3_pt['time'].values])
MOD13_times = np.array([np.datetime64(t) for t in MOD13_pt['time'].values])
MCD43_times = np.array([np.datetime64(t) for t in MCD43_pt['time'].values])


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
mcd19_doy = pd.to_datetime(MCD19_QC1_times).dayofyear
mod13_doy = pd.to_datetime(MOD13_times).dayofyear
mcd43_doy = pd.to_datetime(MCD43_times).dayofyear
photospec_doy = Photospec_times_filtered.dt.dayofyear

# Sort data by day of year so you can plot as lines without weird zigzags
mcd19_doy_sorted = mcd19_doy[np.argsort(mcd19_doy)]
mcd19_ndvi_sorted = MCD19_QC1_pt['NDVI'].values[np.argsort(mcd19_doy)]

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

MOD13_color = '#35DCE8'
MCD19_color = '#CF00BD'
MCD43_color = '#E69F00'

# Plot shaded regions for winter periods
winter_periods = [
    # Ground snow cover periods
    (pd.Timestamp('2017-09-25'), pd.Timestamp('2018-05-30'))
]

fig, ax = plt.subplots(figsize=(7, 3))

for start, end in winter_periods:
    ax.axvspan(start, end, color='lightblue', alpha=0.5)

# Plot Photospec NDVI on left y-axis
ax.scatter(photospec_doy, Photospec_df_filtered['NDVI'], label='Photospec', color='black', s=10, alpha=0.7)
# ax.scatter(mod13_doy_sorted, mod13_ndvi_sorted, label='MOD13 NDVI', color=MOD13_color, s=20, alpha=1)
ax.scatter(mcd19_doy_sorted, mcd19_ndvi_sorted, label=f'MCD19 NDVI\n({QC1_descr})', color=MCD19_color, s=20, alpha=1)
# ax.scatter(mcd43_doy_sorted, mcd43_ndvi_sorted, label='MCD43 NDVI', color=MCD43_color, s=20, alpha=1)


# Adjust limits and legend
ax.set_ylabel('NDVI')
ax.set_xticks(month_starts)
ax.set_xticklabels(month_labels)


# Combine legends using Patch
handles = [
    # Patch(color='lightblue', label=f'SNODAS snow\ndepth > {snow_depth_thresh_mm} mm', alpha=0.5),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label='Photospec'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=MCD19_color, markersize=8, label='MCD19'),
    # plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=MOD13_color, markersize=8, label='MOD13'),
    # plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=MCD43_color, markersize=8, label='MCD43')
]
ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.05, 1))
ax.set_title(site_name)
plt.tight_layout()
plt.show()

#%%

fig, ax = plt.subplots(figsize=(6, 3))

# # Plot shaded region where snow depth exceeds threshold
# ax.axvspan(snow_exceed_start, snow_exceed_end, color='lightblue', label=f'Snow cover', alpha=0.3)
for start, end in winter_periods:
    ax.axvspan(start, end, color='lightblue', alpha=0.5)

# Plot Photospec NDVI on left y-axis
ax.scatter(Photospec_times_filtered, Photospec_df_filtered['NDVI'], label='Photospec', color='black', s=10, alpha=0.7)
# ax.scatter(MOD13_times, MOD13_pt['NDVI'], label='MOD13 NDVI', color=MOD13_color, s=28, alpha=1)
ax.scatter(MCD19_QC1_times, MCD19_QC1_pt['NDVI'].values, label=f'MCD19 NDVI\n({QC1_descr})', color=MCD19_color, s=28, alpha=1)
# ax.scatter(MCD43_times, MCD43_pt['NDVI'], label='MCD43 NDVI', color=MCD43_color, s=28, alpha=1)

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

for start, end in winter_periods:
    ax.axvspan(start, end, color='lightblue', alpha=0.5)
    
# ax.axvspan(first_highsnow_doy - 15, last_highsnow_doy + 15, color='lightblue', label=f'SNODAS snow depth > {snow_depth_thresh_mm} mm', alpha=0.5)
ax.scatter(photospec_doy, Photospec_df_filtered['NDVI'], label='Photospec', color='black', s=10, alpha=0.7)
# ax.scatter(mod13_doy_sorted, mod13_ndvi_sorted, label='MOD13 NDVI', color=MOD13_color, s=28, alpha=1)
ax.scatter(mcd19_doy_sorted, mcd19_ndvi_sorted, label='MCD19 NDVI', color=MCD19_color, s=28, alpha=1)
# ax.scatter(mcd43_doy_sorted, mcd43_ndvi_sorted, label='MCD43 NDVI', color=MCD43_color, s=28, alpha=1)
ax.set_ylabel('NDVI')
ax.set_xticks(month_starts)
ax.set_xticklabels(month_labels)
ax.set_title(site_name)
handles = [
    # Patch(color='lightblue', label=f'SNODAS snow\ndepth > {snow_depth_thresh_mm} mm', alpha=0.5),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label='Photospec'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=MCD19_color, markersize=8, label='MCD19'),
    # plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=MOD13_color, markersize=8, label='MOD13'),
    # plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=MCD43_color, markersize=8, label='MCD43')
]
ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.05, 1))

# --- Bottom subplot: CCI comparison ---
ax2 = axes[1]
# ax2.axvspan(first_highsnow_doy - 15, last_highsnow_doy + 15, color='lightblue', alpha=0.5)
for start, end in winter_periods:
    ax2.axvspan(start, end, color='lightblue', alpha=0.5)
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

# Plot shaded region where snow depth exceeds threshold
for start, end in winter_periods:
    ax.axvspan(start, end, color='lightblue', alpha=0.5)

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
    Patch(color = 'lightblue', label=f'Estimated snow cover period', alpha=0.3),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label='PhotoSpec'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=MCD19_color, markersize=8, label='MCD19 ("Clear" only)'),
]
ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.24, 1))

plt.tight_layout()
plt.show()


#%%



# Calculate day of year for MCD19_CloudFree_times and Photospec_times_filtered
mcd19_doy = pd.to_datetime(MCD19_QC1_times).dayofyear
photospec_doy = Photospec_times_filtered.dt.dayofyear

fig, axes = plt.subplots(3, 1, figsize=(10.5, 7), sharex=True, sharey=True)
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
    for start, end in winter_periods:
        ax.axvspan(start, end, color='lightblue', alpha=0.5)
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
            Patch(color='lightblue', label=f'Estimated snow cover period', alpha=0.3),
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
    # # Plot shaded region where snow depth exceeds threshold
    for start, end in winter_periods:
        ax.axvspan(start, end, color='lightblue', alpha=0.5)

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
MCD19_QC1_snowfilt = MCD19_QC1_snowfilt.where(snodas_xr_MODtime <= snow_depth_thresh_mm)

MCD19_QC2_snowfilt = MCD19_QC2_pt.copy()
MCD19_QC2_snowfilt = MCD19_QC2_snowfilt.where(snodas_xr_MODtime <= snow_depth_thresh_mm)

MCD19_QC3_snowfilt = MCD19_QC3_pt.copy()
MCD19_QC3_snowfilt = MCD19_QC3_snowfilt.where(snodas_xr_MODtime <= snow_depth_thresh_mm)




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


plt.figure(figsize=(3.5, 3.4))
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
plt.figure(figsize=(3.5, 3.4))
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
plt.ylim(mcd19_cci_min - (mcd19_cci_range*0.05), mcd19_cci_max + (mcd19_cci_range*0.05))
plt.xlabel('Photospec CCI')
plt.ylabel('MCD19 CCI')
plt.title(QC2_descr)
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


plt.figure(figsize=(3.5, 3.6))
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
