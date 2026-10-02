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
site_name = 'OSBS'
plot_title = 'US-xSB'

MODIS_tile = 'h10v06'

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

Photospec_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/data/PhotoSpec'
Photospec_file = os.path.join(Photospec_dir, f'PhotoSpec_{site_name}.csv')

# MCD19 mean overpass time at OSBS is 11:50 local mean solar time (LMST) = 12:18 EST (PhotoSpec clock).
# Select PhotoSpec times within +/- 1 hr of overpass, rounded to the nearest half hour: [start, end)
overpass_window = ('11:30', '13:30')
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

    Photospec_df = pd.read_csv(Photospec_file)
    print("Photospec_df columns:", Photospec_df.columns.tolist())

    # Remove any data from Photospec_df where NDVI is below 0.5
    Photospec_df = Photospec_df[Photospec_df['NDVI'] >= 0.5].reset_index(drop=True)
    # Convert Photospec timestamps from "M/D/YY HH:MM" format to datetime64
    Photospec_times = pd.to_datetime(Photospec_df['Time'], format='%m/%d/%y %H:%M')

    # Select MCD19 data within the time range of Photospec data
    MCD19_QC1_pt = MCD19_QC1_pt.sel(time=slice(
        Photospec_times.iloc[0] - pd.Timedelta(days=16),
        Photospec_times.iloc[-1] + pd.Timedelta(days=16),
    ))
    MCD19_QC2_pt = MCD19_QC2_pt.sel(time=slice(
        Photospec_times.iloc[0] - pd.Timedelta(days=16),
        Photospec_times.iloc[-1] + pd.Timedelta(days=16),
    ))
    MCD19_QC3_pt = MCD19_QC3_pt.sel(time=slice(
        Photospec_times.iloc[0] - pd.Timedelta(days=16),
        Photospec_times.iloc[-1] + pd.Timedelta(days=16),
    ))
    MCD19_QC4_pt = MCD19_QC4_pt.sel(time=slice(
        Photospec_times.iloc[0] - pd.Timedelta(days=16),
        Photospec_times.iloc[-1] + pd.Timedelta(days=16),
    ))


    MCD19_QC1_times = np.array([np.datetime64(t) for t in MCD19_QC1_pt['time'].values])
    MCD19_QC2_times = np.array([np.datetime64(t) for t in MCD19_QC2_pt['time'].values])
    MCD19_QC3_times = np.array([np.datetime64(t) for t in MCD19_QC3_pt['time'].values])
    MCD19_QC4_times = np.array([np.datetime64(t) for t in MCD19_QC4_pt['time'].values])


    # Filter Photospec_df for MCD19 overpass time +/- 1 hr (see overpass_window)
    mask = Photospec_times.dt.time.between(*overpass_window_times, inclusive='left')
    # Average the overpass-window PhotoSpec obs to daily means (~4 obs/day; 2 for hourly US-NR1)
    Photospec_df_filtered = Photospec_df[mask].assign(Time=Photospec_times[mask].dt.normalize())
    Photospec_df_filtered = Photospec_df_filtered.groupby('Time', as_index=False).mean(numeric_only=True)
    Photospec_times_filtered = Photospec_df_filtered['Time']

    # Set x-axis to month abbreviations at the start of each month
    months = np.arange(1, 13)
    month_starts = [pd.Timestamp(f'{composite_years[0]}-{m:02d}-01').dayofyear for m in months]
    month_labels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']


    #%%
    # Set global matplotlib font sizes for axes and legends
    plt.rcParams.update({
       'axes.labelsize': 15,
       'axes.titlesize': 19,
       'xtick.labelsize': 15,
       'ytick.labelsize': 15,
       'legend.fontsize': 13
    })


    MCD19_QC1_color = '#7AC7FF'
    MCD19_QC2_color =   '#DB830B' 
    MCD19_QC3_color = '#AF70FF'   
    MCD19_QC4_color =  '#BF2665' 


    # Plot shaded regions for winter periods
    winter_periods = [
        # Ground snow cover periods
    ]




    ############################################
    ### QC ANALYSIS DELIVERABLE 1 - NDVI time series v2 (two axes)
    ############################################


    # Calculate day of year for MCD19_CloudFree_times and Photospec_times_filtered

    ndvi_arrays = [
        MCD19_QC1_pt['NDVI'].values,
        MCD19_QC2_pt['NDVI'].values,
        MCD19_QC3_pt['NDVI'].values,
        MCD19_QC4_pt['NDVI'].values,
    ]
    mcd19_ndvi_min = np.nanmin(np.concatenate(ndvi_arrays))
    mcd19_ndvi_max = np.nanmax(np.concatenate(ndvi_arrays))
    mcd19_ndvi_range = mcd19_ndvi_max - mcd19_ndvi_min


    fig, ax = plt.subplots(figsize=(7, 3))

    for start, end in winter_periods:
        ax.axvspan(start, end, color='lightblue', alpha=0.5)
    ax.scatter(Photospec_times_filtered, Photospec_df_filtered['NDVI'], color='black', s=10, alpha=0.7)
    ax2 = ax.twinx()


    ax2.plot(MCD19_QC1_times, MCD19_QC1_pt['NDVI'].values, color=MCD19_QC1_color, marker = 'o', markersize=8, linewidth=3.5, alpha=1, markeredgecolor='black', markeredgewidth=0.5)
    ax2.plot(MCD19_QC2_times, MCD19_QC2_pt['NDVI'].values, color=MCD19_QC2_color, marker = 'o', markersize=8, linewidth=3.5, alpha=1, markeredgecolor='black', markeredgewidth=0.5)
    ax2.plot(MCD19_QC3_times, MCD19_QC3_pt['NDVI'].values, color=MCD19_QC3_color, marker = 'o', markersize=8, linewidth=3.5, alpha=1, markeredgecolor='black', markeredgewidth=0.5)
    ax2.plot(MCD19_QC4_times, MCD19_QC4_pt['NDVI'].values, color=MCD19_QC4_color, marker = 'o', markersize=8, linewidth=3.5, alpha=1, markeredgecolor='black', markeredgewidth=0.5)

    ax2.set_ylim(mcd19_ndvi_min - (mcd19_ndvi_range*0.05), mcd19_ndvi_max + (mcd19_ndvi_range*0.05))
    ax.yaxis.set_major_formatter(FFmt(lambda y, _: f'{y:.1f}'))
    ax2.yaxis.set_major_formatter(FFmt(lambda y, _: f'{y:.1f}'))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax2.yaxis.set_major_locator(MultipleLocator(0.1))

    ax.set_xlim(Photospec_times_filtered.min() - pd.Timedelta(days=10), Photospec_times_filtered.max() + pd.Timedelta(days=10))
    ax.xaxis.set_major_locator(MonthLocator(bymonth=[7, 10], interval=1))
    ax.xaxis.set_major_locator(MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(DateFormatter('%b\n%Y'))
    ax.set_title(f'k)  {plot_title}')
    ax.set_ylabel('NDVI (Tower)')
    ax2.set_ylabel('NDVI (MCD19)')

    plt.tight_layout()
    ndvi_panel_file = os.path.join(panel_dir, f'{site_name}_NDVI_panel.png')
    plt.savefig(ndvi_panel_file, dpi=300, bbox_inches='tight')
    


    #%%


    ############################################
    ### QC ANALYSIS DELIVERABLE 1 - CCI time series v2 (two axes)
    ############################################


    # Calculate day of year for MCD19_CloudFree_times and Photospec_times_filtered

    CCI_arrays = [
        MCD19_QC1_pt['CCI'].values,
        MCD19_QC2_pt['CCI'].values,
        MCD19_QC3_pt['CCI'].values,
        MCD19_QC4_pt['CCI'].values,
    ]
    mcd19_CCI_min = np.nanmin(np.concatenate(CCI_arrays))
    mcd19_CCI_max = np.nanmax(np.concatenate(CCI_arrays))
    mcd19_CCI_range = mcd19_CCI_max - mcd19_CCI_min


    fig, ax = plt.subplots(figsize=(7, 3))

    for start, end in winter_periods:
        ax.axvspan(start, end, color='lightblue', alpha=0.5)
    ax.scatter(Photospec_times_filtered, Photospec_df_filtered['CCI'], color='black', s=10, alpha=0.7)
    ax2 = ax.twinx()


    ax2.plot(MCD19_QC1_times, MCD19_QC1_pt['CCI'].values, color=MCD19_QC1_color, marker = 'o', markersize=8, linewidth=3.5, alpha=1, markeredgecolor='black', markeredgewidth=0.5)
    ax2.plot(MCD19_QC2_times, MCD19_QC2_pt['CCI'].values, color=MCD19_QC2_color, marker = 'o', markersize=8, linewidth=3.5, alpha=1, markeredgecolor='black', markeredgewidth=0.5)
    ax2.plot(MCD19_QC3_times, MCD19_QC3_pt['CCI'].values, color=MCD19_QC3_color, marker = 'o', markersize=8, linewidth=3.5, alpha=1, markeredgecolor='black', markeredgewidth=0.5)
    ax2.plot(MCD19_QC4_times, MCD19_QC4_pt['CCI'].values, color=MCD19_QC4_color, marker = 'o', markersize=8, linewidth=3.5, alpha=1, markeredgecolor='black', markeredgewidth=0.5)

    ax2.set_ylim(mcd19_CCI_min - (mcd19_CCI_range*0.05), mcd19_CCI_max + (mcd19_CCI_range*0.05))
    ax.yaxis.set_major_formatter(FFmt(lambda y, _: f'{y:.1f}'))
    ax2.yaxis.set_major_formatter(FFmt(lambda y, _: f'{y:.1f}'))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax2.yaxis.set_major_locator(MultipleLocator(0.1))

    ax.set_xlim(Photospec_times_filtered.min() - pd.Timedelta(days=10), Photospec_times_filtered.max() + pd.Timedelta(days=10))
    ax.xaxis.set_major_locator(MonthLocator(bymonth=[7, 10], interval=1))
    ax.xaxis.set_major_locator(MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(DateFormatter('%b\n%Y'))
    ax.set_title(f'l)  {plot_title}')
    ax.set_ylabel('CCI (Tower)')
    ax2.set_ylabel('CCI (MCD19)')

    plt.tight_layout()
    cci_panel_file = os.path.join(panel_dir, f'{site_name}_CCI_panel.png')
    plt.savefig(cci_panel_file, dpi=300, bbox_inches='tight')
    




    #%%
    #~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#
    #~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#

    ### Calculate fit params

    #~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#
    #~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#

    # First, Interpolate Photospec CCI to match MCD19_QC1_pt.time
    photospec_cci_interp = pd.Series(
        np.interp(
            pd.to_datetime(MCD19_QC1_pt.time.values).astype(np.int64),
            Photospec_times_filtered.astype(np.int64),
            Photospec_df_filtered['CCI'].values
        ),
        index=MCD19_QC1_pt.time.values
    )


    # First, Interpolate Photospec CCI to match MCD19_QC1_pt.time
    photospec_ndvi_interp = pd.Series(
        np.interp(
            pd.to_datetime(MCD19_QC1_pt.time.values).astype(np.int64),
            Photospec_times_filtered.astype(np.int64),
            Photospec_df_filtered['NDVI'].values
        ),
        index=MCD19_QC1_pt.time.values
    )


    # Interpolate Photospec values to MCD19 times, but do not bridge long gaps
    # in the source observations.  A gap longer than one composite period is
    # treated as missing rather than being filled by np.interp.
    # (same 16-day rule as the US-UMB/US-Ne3 scripts and 03_final_compare/)
    def interpolate_without_long_nan_gaps(source_times, source_values, target_times,
                                          max_gap_days=16):
        source_times = pd.to_datetime(source_times)
        target_times = pd.to_datetime(target_times)
        source_values = np.asarray(source_values, dtype=float)
        valid = np.isfinite(source_values)

        source_time_ns = source_times.astype(np.int64).to_numpy()
        target_time_ns = target_times.astype(np.int64).to_numpy()
        valid_times = source_time_ns[valid]
        valid_values = source_values[valid]

        result = np.interp(target_time_ns, valid_times, valid_values).astype(float)
        result[(target_times < source_times.iloc[0]) |
               (target_times > source_times.iloc[-1])] = np.nan

        # Mask intervals between valid observations when the intervening source
        # gap (including its endpoints) is longer than the allowed duration.
        max_gap = pd.Timedelta(days=max_gap_days).value
        valid_indices = np.flatnonzero(valid)
        for left, right in zip(valid_indices[:-1], valid_indices[1:]):
            if source_time_ns[right] - source_time_ns[left] > max_gap:
                result[(target_time_ns > source_time_ns[left]) &
                       (target_time_ns < source_time_ns[right])] = np.nan

        return pd.Series(result, index=MCD19_QC1_pt.time.values)


    photospec_cci_interp = interpolate_without_long_nan_gaps(
        Photospec_times_filtered, Photospec_df_filtered['CCI'].values,
        MCD19_QC1_pt.time.values
    )

    photospec_ndvi_interp = interpolate_without_long_nan_gaps(
        Photospec_times_filtered, Photospec_df_filtered['NDVI'].values,
        MCD19_QC1_pt.time.values
    )


    # Set to nan before first and after last valid Photospec time
    first_time = Photospec_times_filtered.iloc[0]
    last_time = Photospec_times_filtered.iloc[-1]
    interp_times = pd.to_datetime(MCD19_QC1_pt.time.values)

    photospec_cci_interp[(interp_times < first_time) | (interp_times > last_time)] = np.nan
    photospec_ndvi_interp[(interp_times < first_time) | (interp_times > last_time)] = np.nan

    # Convert interp_times to dayofyear
    interp_times_doy = pd.to_datetime(interp_times).dayofyear

    # Calculate day of year for MCD19_CloudFree_times and Photospec_times_filtered
    mcd19_doy = pd.to_datetime(MCD19_QC1_pt.time.values).dayofyear
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
    

    #%%



    ############################################
    ### QC1
    ############################################

    # Blank table for fit statistics by QC option and element.
    fit_stats_df = pd.DataFrame(
        index=pd.MultiIndex.from_tuples(
            [
                ('QC1', QC1_descr, 'CCI'),
                ('QC1', QC1_descr, 'NDVI'),
                ('QC2', QC2_descr, 'CCI'),
                ('QC2', QC2_descr, 'NDVI'),
                ('QC3', QC3_descr, 'CCI'),
                ('QC3', QC3_descr, 'NDVI'),
                ('QC4', QC4_descr, 'CCI'),
                ('QC4', QC4_descr, 'NDVI'),
            ],
            names=['QC', 'QC_descr', 'element'],
        ),
        columns=['R2', 'bias', 'CRMSE', 'Spearman_r', 'Spearman_pval'],
        dtype=float,
    )

    # Only compare where both are valid (not nan)
    valid_mask = (~np.isnan(MCD19_QC1_pt['CCI'].values)) & (~np.isnan(photospec_cci_interp.values))
    # Compute R^2 for the scatter of interpolated Photospec CCI vs filtered MCD19 CCI
    if np.sum(valid_mask) > 1:
        slope, intercept, r_value, p_value, std_err = linregress(
            photospec_cci_interp.values[valid_mask],
            MCD19_QC1_pt['CCI'].values[valid_mask]
        )
        # Spearman rank correlation
        spear_r, spear_p = spearmanr(photospec_cci_interp.values[valid_mask], MCD19_QC1_pt['CCI'].values[valid_mask])

        # Centered (unbiased) RMSE (CRMSE): removes mean bias
        x = photospec_cci_interp.values[valid_mask]
        y = MCD19_QC1_pt['CCI'].values[valid_mask]
        crmse = np.sqrt(np.nanmean(((y - np.nanmean(y)) - (x - np.nanmean(x)))**2))
        bias = np.nanmean(y) - np.nanmean(x)
        cci_fit_stats = [r_value**2, spear_r, crmse, bias]

        fit_stats_df.loc[('QC1', QC1_descr, 'CCI'), :] = [r_value**2, bias, crmse, spear_r, spear_p]
        print(f"MCD19 QC1 ({legend_descr_QC1}) vs PhotoSpec CCI Fit parameters (interp): R^2={r_value**2:.3f}, bias={bias:.3f}, CRMSE={crmse:.3f}, Spearman r={spear_r:.3f}, p-value={spear_p:.3e}")
    else:
        print("Not enough valid data for regression (interp).")



    # Only compare where both are valid (not nan)
    valid_mask = (~np.isnan(MCD19_QC1_pt['NDVI'].values)) & (~np.isnan(photospec_ndvi_interp.values))
    # Compute R^2 for the scatter of interpolated Photospec NDVI vs filtered MCD19 NDVI
    if np.sum(valid_mask) > 1:
        slope, intercept, r_value, p_value, std_err = linregress(
            photospec_ndvi_interp.values[valid_mask],
            MCD19_QC1_pt['NDVI'].values[valid_mask]
        )
        # Spearman rank correlation
        spear_r, spear_p = spearmanr(photospec_ndvi_interp.values[valid_mask], MCD19_QC1_pt['NDVI'].values[valid_mask])

        # Centered (unbiased) RMSE (CRMSE): removes mean bias
        x = photospec_ndvi_interp.values[valid_mask]
        y = MCD19_QC1_pt['NDVI'].values[valid_mask]
        crmse = np.sqrt(np.nanmean(((y - np.nanmean(y)) - (x - np.nanmean(x)))**2))
        bias = np.nanmean(y) - np.nanmean(x)
        ndvi_fit_stats = [r_value**2, spear_r, crmse, bias]

        fit_stats_df.loc[('QC1', QC1_descr, 'NDVI'), :] = [r_value**2, bias, crmse, spear_r, spear_p]
        print(f"MCD19 QC1 ({legend_descr_QC1}) vs PhotoSpec NDVI Fit parameters (interp): R^2={r_value**2:.3f}, bias={bias:.3f}, CRMSE={crmse:.3f}, Spearman r={spear_r:.3f}, p-value={spear_p:.3e}")
    else:
        print("Not enough valid data for regression (interp).")




    ############################################
    ### QC2
    ############################################


    # Only compare where both are valid (not nan)
    valid_mask = (~np.isnan(MCD19_QC2_pt['CCI'].values)) & (~np.isnan(photospec_cci_interp.values))
    # Compute R^2 for the scatter of interpolated Photospec CCI vs filtered MCD19 CCI
    if np.sum(valid_mask) > 1:
        slope, intercept, r_value, p_value, std_err = linregress(
            photospec_cci_interp.values[valid_mask],
            MCD19_QC2_pt['CCI'].values[valid_mask]
        )
        # Spearman rank correlation
        spear_r, spear_p = spearmanr(photospec_cci_interp.values[valid_mask], MCD19_QC2_pt['CCI'].values[valid_mask])

        # Centered (unbiased) RMSE (CRMSE): removes mean bias
        x = photospec_cci_interp.values[valid_mask]
        y = MCD19_QC2_pt['CCI'].values[valid_mask]
        crmse = np.sqrt(np.nanmean(((y - np.nanmean(y)) - (x - np.nanmean(x)))**2))
        bias = np.nanmean(y) - np.nanmean(x)
        cci_fit_stats = [r_value**2, spear_r, crmse, bias]

        fit_stats_df.loc[('QC2', QC2_descr, 'CCI'), :] = [r_value**2, bias, crmse, spear_r, spear_p]
        print(f"MCD19 QC2 ({legend_descr_QC2}) vs PhotoSpec CCI Fit parameters (interp): R^2={r_value**2:.3f}, bias={bias:.3f}, CRMSE={crmse:.3f}, Spearman r={spear_r:.3f}, p-value={spear_p:.3e}")
    else:
        print("Not enough valid data for regression (interp).")



    # Only compare where both are valid (not nan)
    valid_mask = (~np.isnan(MCD19_QC2_pt['NDVI'].values)) & (~np.isnan(photospec_ndvi_interp.values))
    # Compute R^2 for the scatter of interpolated Photospec NDVI vs filtered MCD19 NDVI
    if np.sum(valid_mask) > 1:
        slope, intercept, r_value, p_value, std_err = linregress(
            photospec_ndvi_interp.values[valid_mask],
            MCD19_QC2_pt['NDVI'].values[valid_mask]
        )
        # Spearman rank correlation
        spear_r, spear_p = spearmanr(photospec_ndvi_interp.values[valid_mask], MCD19_QC2_pt['NDVI'].values[valid_mask])

        # Centered (unbiased) RMSE (CRMSE): removes mean bias
        x = photospec_ndvi_interp.values[valid_mask]
        y = MCD19_QC2_pt['NDVI'].values[valid_mask]
        crmse = np.sqrt(np.nanmean(((y - np.nanmean(y)) - (x - np.nanmean(x)))**2))
        bias = np.nanmean(y) - np.nanmean(x)
        ndvi_fit_stats = [r_value**2, spear_r, crmse, bias]

        fit_stats_df.loc[('QC2', QC2_descr, 'NDVI'), :] = [r_value**2, bias, crmse, spear_r, spear_p]
        print(f"MCD19 QC2 ({legend_descr_QC2}) vs PhotoSpec NDVI Fit parameters (interp): R^2={r_value**2:.3f}, bias={bias:.3f}, CRMSE={crmse:.3f}, Spearman r={spear_r:.3f}, p-value={spear_p:.3e}")
    else:
        print("Not enough valid data for regression (interp).")




    ############################################
    ### QC3
    ############################################

    # Only compare where both are valid (not nan)
    valid_mask = (~np.isnan(MCD19_QC3_pt['CCI'].values)) & (~np.isnan(photospec_cci_interp.values))
    # Compute R^2 for the scatter of interpolated Photospec CCI vs filtered MCD19 CCI
    if np.sum(valid_mask) > 1:
        slope, intercept, r_value, p_value, std_err = linregress(
            photospec_cci_interp.values[valid_mask],
            MCD19_QC3_pt['CCI'].values[valid_mask]
        )
        # Spearman rank correlation
        spear_r, spear_p = spearmanr(photospec_cci_interp.values[valid_mask], MCD19_QC3_pt['CCI'].values[valid_mask])

        # Centered (unbiased) RMSE (CRMSE): removes mean bias
        x = photospec_cci_interp.values[valid_mask]
        y = MCD19_QC3_pt['CCI'].values[valid_mask]
        crmse = np.sqrt(np.nanmean(((y - np.nanmean(y)) - (x - np.nanmean(x)))**2))
        bias = np.nanmean(y) - np.nanmean(x)
        cci_fit_stats = [r_value**2, spear_r, crmse, bias]

        fit_stats_df.loc[('QC3', QC3_descr, 'CCI'), :] = [r_value**2, bias, crmse, spear_r, spear_p]
        print(f"MCD19 QC3 ({legend_descr_QC3}) vs PhotoSpec CCI Fit parameters (interp): R^2={r_value**2:.3f}, bias={bias:.3f}, CRMSE={crmse:.3f}, Spearman r={spear_r:.3f}, p-value={spear_p:.3e}")
    else:
        print("Not enough valid data for regression (interp).")



    # Only compare where both are valid (not nan)
    valid_mask = (~np.isnan(MCD19_QC3_pt['NDVI'].values)) & (~np.isnan(photospec_ndvi_interp.values))
    # Compute R^2 for the scatter of interpolated Photospec NDVI vs filtered MCD19 NDVI
    if np.sum(valid_mask) > 1:
        slope, intercept, r_value, p_value, std_err = linregress(
            photospec_ndvi_interp.values[valid_mask],
            MCD19_QC3_pt['NDVI'].values[valid_mask]
        )
        # Spearman rank correlation
        spear_r, spear_p = spearmanr(photospec_ndvi_interp.values[valid_mask], MCD19_QC3_pt['NDVI'].values[valid_mask])

        # Centered (unbiased) RMSE (CRMSE): removes mean bias
        x = photospec_ndvi_interp.values[valid_mask]
        y = MCD19_QC3_pt['NDVI'].values[valid_mask]
        crmse = np.sqrt(np.nanmean(((y - np.nanmean(y)) - (x - np.nanmean(x)))**2))
        bias = np.nanmean(y) - np.nanmean(x)
        ndvi_fit_stats = [r_value**2, spear_r, crmse, bias]

        fit_stats_df.loc[('QC3', QC3_descr, 'NDVI'), :] = [r_value**2, bias, crmse, spear_r, spear_p]
        print(f"MCD19 QC3 ({legend_descr_QC3}) vs PhotoSpec NDVI Fit parameters (interp): R^2={r_value**2:.3f}, bias={bias:.3f}, CRMSE={crmse:.3f}, Spearman r={spear_r:.3f}, p-value={spear_p:.3e}")
    else:
        print("Not enough valid data for regression (interp).")



    ############################################
    ### QC4
    ############################################

    # Only compare where both are valid (not nan)
    valid_mask = (~np.isnan(MCD19_QC4_pt['CCI'].values)) & (~np.isnan(photospec_cci_interp.values))
    # Compute R^2 for the scatter of interpolated Photospec CCI vs filtered MCD19 CCI
    if np.sum(valid_mask) > 1:
        slope, intercept, r_value, p_value, std_err = linregress(
            photospec_cci_interp.values[valid_mask],
            MCD19_QC4_pt['CCI'].values[valid_mask]
        )
        # Spearman rank correlation
        spear_r, spear_p = spearmanr(photospec_cci_interp.values[valid_mask], MCD19_QC4_pt['CCI'].values[valid_mask])

        # Centered (unbiased) RMSE (CRMSE): removes mean bias
        x = photospec_cci_interp.values[valid_mask]
        y = MCD19_QC4_pt['CCI'].values[valid_mask]
        crmse = np.sqrt(np.nanmean(((y - np.nanmean(y)) - (x - np.nanmean(x)))**2))
        bias = np.nanmean(y) - np.nanmean(x)
        cci_fit_stats = [r_value**2, spear_r, crmse, bias]

        fit_stats_df.loc[('QC4', QC4_descr, 'CCI'), :] = [r_value**2, bias, crmse, spear_r, spear_p]
        print(f"MCD19 QC4 ({legend_descr_QC4}) vs PhotoSpec CCI Fit parameters (interp): R^2={r_value**2:.3f}, bias={bias:.3f}, CRMSE={crmse:.3f}, Spearman r={spear_r:.3f}, p-value={spear_p:.3e}")
    else:
        print("Not enough valid data for regression (interp).")



    # Only compare where both are valid (not nan)
    valid_mask = (~np.isnan(MCD19_QC4_pt['NDVI'].values)) & (~np.isnan(photospec_ndvi_interp.values))
    # Compute R^2 for the scatter of interpolated Photospec NDVI vs filtered MCD19 NDVI
    if np.sum(valid_mask) > 1:
        slope, intercept, r_value, p_value, std_err = linregress(
            photospec_ndvi_interp.values[valid_mask],
            MCD19_QC4_pt['NDVI'].values[valid_mask]
        )
        # Spearman rank correlation
        spear_r, spear_p = spearmanr(photospec_ndvi_interp.values[valid_mask], MCD19_QC4_pt['NDVI'].values[valid_mask])

        # Centered (unbiased) RMSE (CRMSE): removes mean bias
        x = photospec_ndvi_interp.values[valid_mask]
        y = MCD19_QC4_pt['NDVI'].values[valid_mask]
        crmse = np.sqrt(np.nanmean(((y - np.nanmean(y)) - (x - np.nanmean(x)))**2))
        bias = np.nanmean(y) - np.nanmean(x)
        ndvi_fit_stats = [r_value**2, spear_r, crmse, bias]

        fit_stats_df.loc[('QC4', QC4_descr, 'NDVI'), :] = [r_value**2, bias, crmse, spear_r, spear_p]
        print(f"MCD19 QC4 ({legend_descr_QC4}) vs PhotoSpec NDVI Fit parameters (interp): R^2={r_value**2:.3f}, bias={bias:.3f}, CRMSE={crmse:.3f}, Spearman r={spear_r:.3f}, p-value={spear_p:.3e}")
    else:
        print("Not enough valid data for regression (interp).")



    fit_stats_outfile = os.path.join(out_stats_dir, f'MCD19_v_PhotoSpec_fit_stats_{out_stats_descr}_{site_name}.csv')
    fit_stats_df.to_csv(fit_stats_outfile)


if __name__ == "__main__":
    main()
