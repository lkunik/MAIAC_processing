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
site_name = 'US-Ne3'
plot_title = site_name

MODIS_tile = 'h10v04'

# Compare AOD and adjacency masks:
# 1: CloudFree_AllAOD_AllAdj
# 2: CloudFree_LowAOD_AllAdj

# 3: CloudFree_AllAOD_NotSurround
# 4: CloudFree_LowAOD_NotSurround

# 5: CloudFree_AllAOD_ClearAdj
# 6: CloudFree_LowAOD_ClearAdj

QC1_descr = "CloudFree_AllAOD_AllAdj"
QC2_descr = "CloudFree_LowAOD_AllAdj"
QC3_descr = "CloudFree_AllAOD_ClearAdj"
QC4_descr = "CloudFree_LowAOD_ClearAdj"

legend_descr_QC1 = "AllAOD, AllAdj"
legend_descr_QC2 = "LowAOD, AllAdj"
legend_descr_QC3 = "AllAOD, ClearAdj"
legend_descr_QC4 = "LowAOD, ClearAdj"


out_stats_descr = "QC_Compare"

dat_pt_basedir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/data/QC_PhotoSpec_sites_backup/'
dat_pt_basedir_QC1 = os.path.join(dat_pt_basedir, f'CV-MVC/pre-fill/QCfilt_{QC1_descr}')
dat_pt_basedir_QC2 = os.path.join(dat_pt_basedir, f'CV-MVC/pre-fill/QCfilt_{QC2_descr}')
dat_pt_basedir_QC3 = os.path.join(dat_pt_basedir, f'CV-MVC/pre-fill/QCfilt_{QC3_descr}')
dat_pt_basedir_QC4 = os.path.join(dat_pt_basedir, f'CV-MVC/pre-fill/QCfilt_{QC4_descr}')

FloX_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/data/FloX/US-Ne3/'
FloX_2018_file = os.path.join(FloX_dir, f'DFlox_SIF_VIs_2018.csv')
FloX_2019_file = os.path.join(FloX_dir, f'DFlox_SIF_VIs_2019.csv')

# MCD19 mean overpass time at US-Ne3 is 12:04 local mean solar time (LMST). FloX DateTime is UTC, so
# convert to LMST (UTC + lon/15) before selecting FloX times within +/- 1 hr of overpass
FloX_utc_to_LMST = pd.Timedelta(hours=-96.4397 / 15)
overpass_window = ('11:04', '13:04')
overpass_window_times = tuple(pd.Timestamp(t).time() for t in overpass_window)


out_stats_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/output/CVMVC_QC_regression_stats/'
os.makedirs(out_stats_dir, exist_ok=True)

plot_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/plots/CVMVC_QC_regression_stats/'
os.makedirs(plot_dir, exist_ok=True)

panel_dir = os.path.join(plot_dir, 'panels')
os.makedirs(panel_dir, exist_ok=True)

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


def remove_running_window_outliers(df, window='14D', value_columns=('CCI', 'NDVI')):
    """Remove rows containing values outside running 1st/99th percentiles."""
    df = df.sort_values('DateTime').copy()
    indexed = df.set_index('DateTime')
    keep = pd.Series(True, index=indexed.index)

    for column in value_columns:
        if column not in indexed:
            continue
        values = indexed[column]
        lower = values.rolling(window, center=True, min_periods=2).quantile(0.01)
        upper = values.rolling(window, center=True, min_periods=2).quantile(0.99)
        keep &= values.isna() | ((values >= lower) & (values <= upper))

    return indexed.loc[keep.to_numpy()].reset_index()

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
def main():

    # mark start time to keep track of elapsed
    start_total = time.time()


    MCD19_QC1_file = os.path.join(dat_pt_basedir_QC1, f'MCD19_QCfilt_{QC1_descr}_{site_name}.nc')
    MCD19_QC2_file = os.path.join(dat_pt_basedir_QC2, f'MCD19_QCfilt_{QC2_descr}_{site_name}.nc')
    MCD19_QC3_file = os.path.join(dat_pt_basedir_QC3, f'MCD19_QCfilt_{QC3_descr}_{site_name}.nc')
    MCD19_QC4_file = os.path.join(dat_pt_basedir_QC4, f'MCD19_QCfilt_{QC4_descr}_{site_name}.nc')


    MCD19_QC1_pt = xr.open_dataset(MCD19_QC1_file)
    MCD19_QC2_pt = xr.open_dataset(MCD19_QC2_file)
    MCD19_QC3_pt = xr.open_dataset(MCD19_QC3_file)
    MCD19_QC4_pt = xr.open_dataset(MCD19_QC4_file)


    #%%

    FloX_df_2018 = pd.read_csv(FloX_2018_file)


    print("FloX_df_2018 columns:", FloX_df_2018.columns.tolist())
    # print(f'FloX_df_2018 rows (before filtering for valid datetimes): {len(FloX_df_2018)}')
    FloX_df_2018 = FloX_df_2018[FloX_df_2018['DoY'].between(1, 366)]
    FloX_df_2018['DateTime'] = pd.to_datetime(FloX_df_2018['DateTime'], format='%m/%d/%y %H:%M')
    FloX_df_2018 = FloX_df_2018[FloX_df_2018['DateTime'].between('2018-01-01', '2018-12-31')]
    FloX_df_2018 = FloX_df_2018[
        (FloX_df_2018['DateTime'] + FloX_utc_to_LMST).dt.time.between(*overpass_window_times)
    ]

    FloX_df_2018 = FloX_df_2018[FloX_df_2018['NDVI'].between(0, 1)]
    FloX_df_2018 = remove_running_window_outliers(FloX_df_2018)




    # print(f'FloX_df_2018 rows (after filtering for valid datetimes): {len(FloX_df_2018)}\n\n\n')

    FloX_df_2019 = pd.read_csv(FloX_2019_file)
    # print(f'FloX_df_2019 rows (before filtering for valid datetimes): {len(FloX_df_2019)}')
    FloX_df_2019['DoY'] = FloX_df_2019['DoY'] - 365
    FloX_df_2019 = FloX_df_2019[FloX_df_2019['DoY'].between(1, 366)]
    FloX_df_2019['DateTime'] = pd.to_datetime(FloX_df_2019['DateTime'], format='%m/%d/%y %H:%M')
    FloX_df_2019 = FloX_df_2019[FloX_df_2019['DateTime'].between('2019-01-01', '2019-12-31')]
    FloX_df_2019 = FloX_df_2019[
        (FloX_df_2019['DateTime'] + FloX_utc_to_LMST).dt.time.between(*overpass_window_times)
    ]
    FloX_df_2019 = FloX_df_2019[FloX_df_2019['NDVI'].between(0, 1)]
    FloX_df_2019 = remove_running_window_outliers(FloX_df_2019)
    # print(f'FloX_df_2019 rows (after filtering for valid datetimes): {len(FloX_df_2019)}')


    FloX_df_filtered = pd.concat([FloX_df_2018, FloX_df_2019], ignore_index=True)

    # Aggregate all numeric FloX columns to daily mean values.
    FloX_df_filtered['Date'] = pd.to_datetime(FloX_df_filtered['DateTime']).dt.normalize()
    FloX_df_filtered = (
        FloX_df_filtered.groupby('Date', as_index=False)
        .mean(numeric_only=True)
        .rename(columns={'Date': 'DateTime'})
    )


    # Convert FloX timestamps from "M/D/YY HH:MM" format to datetime64
    FloX_times_filtered = pd.to_datetime(FloX_df_filtered['DateTime'], format='%m/%d/%y %H:%M')

    # Select MCD19 data within the time range of FloX data
    MCD19_QC1_pt = MCD19_QC1_pt.sel(time=slice(
        FloX_times_filtered.iloc[0] - pd.Timedelta(days=16),
        FloX_times_filtered.iloc[-1] + pd.Timedelta(days=16),
    ))
    MCD19_QC2_pt = MCD19_QC2_pt.sel(time=slice(
        FloX_times_filtered.iloc[0] - pd.Timedelta(days=16),
        FloX_times_filtered.iloc[-1] + pd.Timedelta(days=16),
    ))
    MCD19_QC3_pt = MCD19_QC3_pt.sel(time=slice(
        FloX_times_filtered.iloc[0] - pd.Timedelta(days=16),
        FloX_times_filtered.iloc[-1] + pd.Timedelta(days=16),
    ))
    MCD19_QC4_pt = MCD19_QC4_pt.sel(time=slice(
        FloX_times_filtered.iloc[0] - pd.Timedelta(days=16),
        FloX_times_filtered.iloc[-1] + pd.Timedelta(days=16),
    ))


    MCD19_QC1_times = np.array([np.datetime64(t) for t in MCD19_QC1_pt['time'].values])
    MCD19_QC2_times = np.array([np.datetime64(t) for t in MCD19_QC2_pt['time'].values])
    MCD19_QC3_times = np.array([np.datetime64(t) for t in MCD19_QC3_pt['time'].values])
    MCD19_QC4_times = np.array([np.datetime64(t) for t in MCD19_QC4_pt['time'].values])


    # Set x-axis to month abbreviations at the start of each month
    months = np.arange(1, 13)
    month_starts = [pd.Timestamp(f'{composite_years[0]}-{m:02d}-01').dayofyear for m in months]
    month_labels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']


    #%%
    # Set global matplotlib font sizes for axes and legends
    plt.rcParams.update({
       'axes.labelsize': 17,
       'axes.titlesize': 19,
       'xtick.labelsize': 17,
       'ytick.labelsize': 17,
       'legend.fontsize': 13
    })


    MCD19_QC1_color = '#7AC7FF'
    MCD19_QC2_color =   '#DB830B' 
    MCD19_QC3_color = '#AF70FF'   
    MCD19_QC4_color =  '#BF2665' 


    # Plot shaded regions for winter periods
    winter_periods = [
        # Ground snow cover periods
        (pd.Timestamp('2017-12-24'), pd.Timestamp('2018-02-25')),
        (pd.Timestamp('2018-11-09'), pd.Timestamp('2019-03-15')),
        (pd.Timestamp('2019-12-15'), pd.Timestamp('2020-02-08'))
    ]




    ############################################
    ### QC ANALYSIS DELIVERABLE 1 - AOD Compare: Number of Observations time series
    ############################################



    # legend_descr_QC1 = "AllAOD, AllAdj"
    # legend_descr_QC2 = "LowAOD, AllAdj"
    # legend_descr_QC3 = "AllAOD, ClearAdj"
    # legend_descr_QC4 = "LowAOD, ClearAdj"


    qc_colors = [MCD19_QC3_color, MCD19_QC4_color]
    plt.rcParams['hatch.linewidth'] = 2  # controls hatch stroke width, independent of bar edge linewidth


    nobs_stack = np.vstack([
        MCD19_QC3_pt['valid_obs_count'].values,
        MCD19_QC4_pt['valid_obs_count'].values,
    ])
    times = MCD19_QC1_times  # common time axis

    mcd19_nobs_max = np.nanmax(nobs_stack)

    composite_indices = np.arange(len(MCD19_QC1_pt.time.values))
    normalized_days = composite_indices * 16

    # Map actual dates to normalized axis for snow cover shading
    actual_times = pd.to_datetime(MCD19_QC1_times)


    fig, ax = plt.subplots(figsize=(7, 3.2))

    for start, end in winter_periods:
        snow_start_idx = np.argmin(np.abs((actual_times - pd.to_datetime(start)).total_seconds()))
        snow_end_idx = np.argmin(np.abs((actual_times - pd.to_datetime(end)).total_seconds()))

        snow_start_norm = snow_start_idx * 16
        snow_end_norm = snow_end_idx * 16

        if snow_start_norm == 0:
            snow_start_norm = -50
        if snow_end_norm == (len(actual_times)-1)*16:
            snow_end_norm = (len(actual_times)*16) + 50
        ax.axvspan(snow_start_norm, snow_end_norm, facecolor='#A0B7BD', edgecolor='#8DB9C9', hatch='///', alpha=0.25)


    #     ax.axvspan(start, end, facecolor='#A0B7BD', edgecolor='#8DB9C9', hatch='///', alpha=0.25, zorder=0)


    vals_for_rank = np.where(np.isnan(nobs_stack), -np.inf, nobs_stack)
    order = np.argsort(-vals_for_rank, axis=0)  # order[0] = index of larger series per timestep

    sorted_vals = np.take_along_axis(vals_for_rank, order, axis=0)
    tie_mask_sorted = np.zeros_like(sorted_vals, dtype=bool)
    tie_mask_sorted[:-1] |= (sorted_vals[:-1] == sorted_vals[1:]) & np.isfinite(sorted_vals[:-1])
    tie_mask_sorted[1:]  |= (sorted_vals[1:] == sorted_vals[:-1]) & np.isfinite(sorted_vals[1:])

    bar_width = 14
    n_time = nobs_stack.shape[1]
    qc_colors_arr = np.array(qc_colors)

    for rank in range(nobs_stack.shape[0]):
        series_idx = order[rank]                     # which QC series holds this rank, per timestep
        other_idx = 1 - series_idx                    # the *other* series (only valid for exactly 2 series)

        heights = nobs_stack[series_idx, np.arange(n_time)]
        colors = qc_colors_arr[series_idx]
        other_colors = qc_colors_arr[other_idx]
        hatch_on = tie_mask_sorted[rank]

        bars = ax.bar(normalized_days, heights, color=colors, width=bar_width, alpha=1, zorder=rank + 1)
        for bar_patch, on, edge_color in zip(bars, hatch_on, other_colors):
            if on:
                bar_patch.set_hatch('///')
                bar_patch.set_edgecolor(edge_color)
                bar_patch.set_linewidth(0)

    ax.set_ylim(0, mcd19_nobs_max + (mcd19_nobs_max*0.05))
    ax.yaxis.set_major_locator(MultipleLocator(5))

    # ax.set_xlim(Photospec_times.min() - pd.Timedelta(days=10), Photospec_times.max() + pd.Timedelta(days=10))
    # ax.xaxis.set_major_locator(MonthLocator(bymonth=[1, 7]))
    # ax.xaxis.set_major_formatter(DateFormatter('%b\n%Y'))


    # Create x-axis labels: find which composites correspond to month starts
    tick_freq = '6MS'  # '3MS', '6MS', or 'YS' — set per site as needed

    anchor_start = pd.Timestamp(year=actual_times[0].year, month=1, day=1)
    month_dates = pd.date_range(start=anchor_start, end=actual_times[-1], freq=tick_freq)
    month_dates = month_dates[month_dates >= actual_times[0] - pd.Timedelta(days=8)]

    month_indices = [np.argmin(np.abs(actual_times - date)) for date in month_dates]
    month_positions = [idx * 16 for idx in month_indices]
    month_labels = [date.strftime('%b\n%Y') for date in month_dates]
    ax.set_xlim(-16, (len(actual_times)-1)*16)
    ax.set_xticks(month_positions)
    ax.set_xticklabels(month_labels)

    ax.set_title(f'b)  {plot_title}')
    ax.set_ylabel('# of Valid Observations')

    plt.tight_layout()

    nobs_panel_file = os.path.join(panel_dir, f'{site_name}_nobs_AODcompare_panel.png')
    plt.savefig(nobs_panel_file, dpi=300, bbox_inches='tight')


    ############################################
    ### QC ANALYSIS DELIVERABLE 2 - Adjacency Compare: Number of Observations time series
    ############################################

    # legend_descr_QC1 = "AllAOD, AllAdj"
    # legend_descr_QC2 = "LowAOD, AllAdj"
    # legend_descr_QC3 = "AllAOD, ClearAdj"
    # legend_descr_QC4 = "LowAOD, ClearAdj"


    qc_colors = [MCD19_QC2_color, MCD19_QC4_color]

    nobs_stack = np.vstack([
        MCD19_QC2_pt['valid_obs_count'].values,
        MCD19_QC4_pt['valid_obs_count'].values,
    ])
    times = MCD19_QC1_times  # common time axis

    mcd19_nobs_max = np.nanmax(nobs_stack)

    fig, ax = plt.subplots(figsize=(7, 3.2))

    for start, end in winter_periods:
        snow_start_idx = np.argmin(np.abs((actual_times - pd.to_datetime(start)).total_seconds()))
        snow_end_idx = np.argmin(np.abs((actual_times - pd.to_datetime(end)).total_seconds()))

        snow_start_norm = snow_start_idx * 16
        snow_end_norm = snow_end_idx * 16

        if snow_start_norm == 0:
            snow_start_norm = -50
        if snow_end_norm == (len(actual_times)-1)*16:
            snow_end_norm = (len(actual_times)*16) + 50
        ax.axvspan(snow_start_norm, snow_end_norm, facecolor='#A0B7BD', edgecolor='#8DB9C9', hatch='///', alpha=0.25)


    #     ax.axvspan(start, end, facecolor='#A0B7BD', edgecolor='#8DB9C9', hatch='///', alpha=0.25, zorder=0)


    vals_for_rank = np.where(np.isnan(nobs_stack), -np.inf, nobs_stack)
    order = np.argsort(-vals_for_rank, axis=0)  # order[0] = index of larger series per timestep

    sorted_vals = np.take_along_axis(vals_for_rank, order, axis=0)
    tie_mask_sorted = np.zeros_like(sorted_vals, dtype=bool)
    tie_mask_sorted[:-1] |= (sorted_vals[:-1] == sorted_vals[1:]) & np.isfinite(sorted_vals[:-1])
    tie_mask_sorted[1:]  |= (sorted_vals[1:] == sorted_vals[:-1]) & np.isfinite(sorted_vals[1:])

    bar_width = 14
    n_time = nobs_stack.shape[1]
    qc_colors_arr = np.array(qc_colors)

    for rank in range(nobs_stack.shape[0]):
        series_idx = order[rank]                     # which QC series holds this rank, per timestep
        other_idx = 1 - series_idx                    # the *other* series (only valid for exactly 2 series)

        heights = nobs_stack[series_idx, np.arange(n_time)]
        colors = qc_colors_arr[series_idx]
        other_colors = qc_colors_arr[other_idx]
        hatch_on = tie_mask_sorted[rank]

        bars = ax.bar(normalized_days, heights, color=colors, width=bar_width, alpha=1, zorder=rank + 1)
        for bar_patch, on, edge_color in zip(bars, hatch_on, other_colors):
            if on:
                bar_patch.set_hatch('///')
                bar_patch.set_edgecolor(edge_color)
                bar_patch.set_linewidth(0)

    ax.set_ylim(0, mcd19_nobs_max + (mcd19_nobs_max*0.05))
    ax.yaxis.set_major_locator(MultipleLocator(5))

    # ax.set_xlim(Photospec_times.min() - pd.Timedelta(days=10), Photospec_times.max() + pd.Timedelta(days=10))
    # ax.xaxis.set_major_locator(MonthLocator(bymonth=[1, 7]))
    # ax.xaxis.set_major_formatter(DateFormatter('%b\n%Y'))


    # Create x-axis labels: find which composites correspond to month starts
    tick_freq = '6MS'  # '3MS', '6MS', or 'YS' — set per site as needed

    anchor_start = pd.Timestamp(year=actual_times[0].year, month=1, day=1)
    month_dates = pd.date_range(start=anchor_start, end=actual_times[-1], freq=tick_freq)
    month_dates = month_dates[month_dates >= actual_times[0] - pd.Timedelta(days=8)]

    month_indices = [np.argmin(np.abs(actual_times - date)) for date in month_dates]
    month_positions = [idx * 16 for idx in month_indices]
    month_labels = [date.strftime('%b\n%Y') for date in month_dates]
    ax.set_xlim(-16, (len(actual_times)-1)*16)
    ax.set_xticks(month_positions)
    ax.set_xticklabels(month_labels)

    ax.set_title(f'b)  {plot_title}')
    ax.set_ylabel('# of Valid Observations')

    plt.tight_layout()
    nobs_panel_file = os.path.join(panel_dir, f'{site_name}_nobs_Adjcompare_panel.png')
    plt.savefig(nobs_panel_file, dpi=300, bbox_inches='tight')



  
if __name__ == "__main__":
    main()
