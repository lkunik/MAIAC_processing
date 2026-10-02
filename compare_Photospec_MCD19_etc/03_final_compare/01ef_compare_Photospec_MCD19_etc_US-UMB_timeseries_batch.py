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
site_name = 'US-UMB'
plot_title = 'US-UMB'
MODIS_tile = 'h12v04'

QC_descr = "CloudFree_LowAOD_ClearAdj"

MCD19_legend_descr = "CV-MVC" # gets added in parentheses after "MCD19"


dat_pt_basedir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/data/QC_PhotoSpec_sites_backup/'
dat_pt_basedir_QC = os.path.join(dat_pt_basedir, f'CV-MVC/lm-fill/QCfilt_{QC_descr}')

MCD19_QC_file = os.path.join(dat_pt_basedir_QC, f'MCD19_QCfilt_{QC_descr}_{site_name}.nc')


Photospec_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/data/PhotoSpec/US-UMB/'
Photospec_2018_file = os.path.join(Photospec_dir, 'PhotoSpecM1_2018_v20260903.nc')
Photospec_2019_file = os.path.join(Photospec_dir, 'PhotoSpecM1_2019_v20260903.nc')

# MCD19 mean overpass time at US-UMB is 12:21 local mean solar time (LMST) = 13:00 EST (PhotoSpec clock).
# PhotoSpec data are 90-min averages (Hour_of_Day = interval start); HOD index 8 (12:00-13:30 EST)
# is the period that best matches overpass +/- 1 hr
Photospec_HOD_idx = 8
Photospec_HOD_start = pd.Timedelta(hours=12)
overpass_window = ('12:00', '13:30')
overpass_window_times = tuple(pd.Timestamp(t).time() for t in overpass_window)

MOD13_dir = os.path.join(dat_pt_basedir, 'MOD13')
MOD13_file = os.path.join(MOD13_dir, f'MOD13_pt_{site_name}.nc')
MCD43_dir = os.path.join(dat_pt_basedir, 'MCD43')
MCD43_file = os.path.join(MCD43_dir, f'MCD43_pt_{site_name}.nc')

GCOM_dir = os.path.join(dat_pt_basedir, 'GCOM')
GCOM_file = os.path.join(GCOM_dir, f'GCOM_pt_{site_name}.nc')
MOD09_dir = os.path.join(dat_pt_basedir, 'MOD09')
MOD09_file = os.path.join(MOD09_dir, f'MOD09_pt_{site_name}.nc')

out_stats_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/output/CVMVC_QC_regression_stats/final/'
os.makedirs(out_stats_dir, exist_ok=True)

plot_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/plots/CVMVC_QC_regression_stats/final/'
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
def main():

    # mark start time to keep track of elapsed
    start_total = time.time()

    MCD19_QC_pt = xr.open_dataset(MCD19_QC_file)

    MOD13_pt = xr.open_dataset(MOD13_file)
    MCD43_pt = xr.open_dataset(MCD43_file)
    GCOM_pt = xr.open_dataset(GCOM_file)
    MOD09_pt = xr.open_dataset(MOD09_file)


    Photospec_xr_2018 = xr.open_dataset(Photospec_2018_file)
    Photospec_xr_2019 = xr.open_dataset(Photospec_2019_file)


    # Format Photospec data into a single DataFrame with datetime index (like other PS datasets)
    Photospec_df_2018 = Photospec_xr_2018.sel(HOD=Photospec_HOD_idx).to_dataframe().reset_index()

    Photospec_df_2018['Time'] = (
        pd.Timestamp('2018-01-01')
        + pd.to_timedelta(Photospec_df_2018['DOY'].astype(int), unit='D')
        + Photospec_HOD_start
    )


    Photospec_df_2019 = Photospec_xr_2019.sel(HOD=Photospec_HOD_idx).to_dataframe().reset_index()

    Photospec_df_2019['Time'] = (
        pd.Timestamp('2019-01-01')
        + pd.to_timedelta(Photospec_df_2019['DOY'].astype(int), unit='D')
        + Photospec_HOD_start
    )

    Photospec_df = pd.concat([Photospec_df_2018, Photospec_df_2019], ignore_index=True)

    # Remove any data from Photospec_df where NDVI is below 0.6
    Photospec_df = Photospec_df[Photospec_df['NDVI'] >= 0.6].reset_index(drop=True)
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

    # Filter Photospec_df for MCD19 overpass time +/- 1 hr (see overpass_window)
    mask = Photospec_times.dt.time.between(*overpass_window_times, inclusive='left')
    Photospec_df_filtered = Photospec_df[mask].reset_index(drop=True)
    Photospec_times_filtered = Photospec_times[mask].reset_index(drop=True)

    # Set x-axis to month abbreviations at the start of each month
    months = np.arange(1, 13)
    month_starts = [pd.Timestamp(f'{composite_years[0]}-{m:02d}-01').dayofyear for m in months]
    month_labels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']

    #%%
    # Set global matplotlib font sizes for axes and legends
    plt.rcParams.update({
    'axes.labelsize': 18,
    'axes.titlesize': 21,
    'xtick.labelsize': 17,
    'ytick.labelsize': 17,
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
        (pd.Timestamp('2017-11-04'), pd.Timestamp('2018-05-01')),
        (pd.Timestamp('2018-11-09'), pd.Timestamp('2019-04-23')),
        (pd.Timestamp('2019-11-06'), pd.Timestamp('2020-04-24')),
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
    ax.yaxis.set_major_locator(MultipleLocator(0.3))


    ax.set_xlim(Photospec_times_filtered.min() - pd.Timedelta(days=10), Photospec_times_filtered.max() + pd.Timedelta(days=10))
    ax.xaxis.set_major_locator(MonthLocator(bymonth=[7, 10], interval=1))
    ax.xaxis.set_major_locator(MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(DateFormatter('%b\n%Y'))
    # ax.set_title('e)', loc='left')
    ax.set_title(f'e)  {plot_title}', loc='center')
    ax.set_ylabel('NDVI')

    plt.tight_layout()
    ndvi_panel_file = os.path.join(panel_dir, f'{site_name}_NDVI_panel.png')
    plt.savefig(ndvi_panel_file, dpi=300, bbox_inches='tight')



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
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.set_xlim(Photospec_times_filtered.min() - pd.Timedelta(days=10), Photospec_times_filtered.max() + pd.Timedelta(days=10))
    ax.xaxis.set_major_locator(MonthLocator(bymonth=[7, 10], interval=1))
    ax.xaxis.set_major_locator(MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(DateFormatter('%b\n%Y'))
    # ax.set_title('f)', loc='left')
    ax.set_title(f'f)  {plot_title}', loc='center')
    ax.set_ylabel('CCI')

    plt.tight_layout()
    cci_panel_file = os.path.join(panel_dir, f'{site_name}_CCI_panel.png')
    plt.savefig(cci_panel_file, dpi=300, bbox_inches='tight')




    # #%%
    # #~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#
    # #~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#

    # ### Calculate fit params

    # #~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#
    # #~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#


    # # Blank table for fit statistics by QC option and element.
    # fit_stats_df = pd.DataFrame(
    #     index=pd.MultiIndex.from_tuples(
    #         [
    #             ('MCD19', 'CCI'),
    #             ('MCD19', 'NDVI'),
    #             ('MCD43', 'CCI'),
    #             ('MCD43', 'NDVI'),
    #             ('MOD09', 'CCI'),
    #             ('MOD09', 'NDVI'),
    #             ('MOD13', 'CCI'),
    #             ('MOD13', 'NDVI'),
    #             ('GCOM', 'CCI'),
    #             ('GCOM', 'NDVI')
    #         ],
    #         names=['QC', 'element'],
    #     ),
    #     columns=['R2', 'bias', 'CRMSE', 'Spearman_r', 'Spearman_pval'],
    #     dtype=float,
    # )

    # # First, Interpolate Photospec CCI to match MCD19_QC_pt.time
    # photospec_cci_interp = pd.Series(
    #     np.interp(
    #         pd.to_datetime(MCD19_QC_pt.time.values).astype(np.int64),
    #         Photospec_times_filtered.astype(np.int64),
    #         Photospec_df_filtered['CCI'].values
    #     ),
    #     index=MCD19_QC_pt.time.values
    # )


    # # First, Interpolate Photospec CCI to match MCD19_QC_pt.time
    # photospec_ndvi_interp = pd.Series(
    #     np.interp(
    #         pd.to_datetime(MCD19_QC_pt.time.values).astype(np.int64),
    #         Photospec_times_filtered.astype(np.int64),
    #         Photospec_df_filtered['NDVI'].values
    #     ),
    #     index=MCD19_QC_pt.time.values
    # )


    # # Interpolate Photospec values to MCD19 times, but do not bridge long gaps
    # # in the source observations.  A gap longer than one composite period is
    # # treated as missing rather than being filled by np.interp.
    # def interpolate_without_long_nan_gaps(source_times, source_values, target_times,
    #                                     max_gap_days=16):
    #     source_times = pd.to_datetime(source_times)
    #     target_times = pd.to_datetime(target_times)
    #     source_values = np.asarray(source_values, dtype=float)
    #     valid = np.isfinite(source_values)

    #     source_time_ns = source_times.astype(np.int64).to_numpy()
    #     target_time_ns = target_times.astype(np.int64).to_numpy()
    #     valid_times = source_time_ns[valid]
    #     valid_values = source_values[valid]

    #     result = np.interp(target_time_ns, valid_times, valid_values).astype(float)
    #     result[(target_times < source_times.iloc[0]) |
    #         (target_times > source_times.iloc[-1])] = np.nan

    #     # Mask intervals between valid observations when the intervening source
    #     # gap (including its endpoints) is longer than the allowed duration.
    #     max_gap = pd.Timedelta(days=max_gap_days).value
    #     valid_indices = np.flatnonzero(valid)
    #     for left, right in zip(valid_indices[:-1], valid_indices[1:]):
    #         if source_time_ns[right] - source_time_ns[left] > max_gap:
    #             result[(target_time_ns > source_time_ns[left]) &
    #                 (target_time_ns < source_time_ns[right])] = np.nan

    #     return pd.Series(result, index=MCD19_QC_pt.time.values)


    # photospec_cci_interp = interpolate_without_long_nan_gaps(
    #     Photospec_times_filtered, Photospec_df_filtered['CCI'].values,
    #     MCD19_QC_pt.time.values
    # )

    # photospec_ndvi_interp = interpolate_without_long_nan_gaps(
    #     Photospec_times_filtered, Photospec_df_filtered['NDVI'].values,
    #     MCD19_QC_pt.time.values
    # )


    # # Set to nan before first and after last valid Photospec time
    # first_time = Photospec_times_filtered.iloc[0]
    # last_time = Photospec_times_filtered.iloc[-1]
    # interp_times = pd.to_datetime(MCD19_QC_pt.time.values)

    # photospec_cci_interp[(interp_times < first_time) | (interp_times > last_time)] = np.nan
    # photospec_ndvi_interp[(interp_times < first_time) | (interp_times > last_time)] = np.nan

    # # Convert interp_times to dayofyear
    # interp_times_doy = pd.to_datetime(interp_times).dayofyear

    # # Calculate day of year for MCD19_CloudFree_times and Photospec_times_filtered
    # mcd19_doy = pd.to_datetime(MCD19_QC_pt.time.values).dayofyear
    # photospec_doy = Photospec_times_filtered.dt.dayofyear


    # #%%


    # # Only compare where both are valid (not nan)
    # valid_mask = (~np.isnan(MCD19_QC_pt['CCI'].values)) & (~np.isnan(photospec_cci_interp.values))
    # # Compute R^2 for the scatter of interpolated Photospec CCI vs filtered MCD19 CCI
    # if np.sum(valid_mask) > 1:
    #     slope, intercept, r_value, p_value, std_err = linregress(
    #         photospec_cci_interp.values[valid_mask],
    #         MCD19_QC_pt['CCI'].values[valid_mask]
    #     )
    #     # Spearman rank correlation
    #     spear_r, spear_p = spearmanr(photospec_cci_interp.values[valid_mask], MCD19_QC_pt['CCI'].values[valid_mask])

    #     # Centered (unbiased) RMSE (CRMSE): removes mean bias
    #     x = photospec_cci_interp.values[valid_mask]
    #     y = MCD19_QC_pt['CCI'].values[valid_mask]
    #     crmse = np.sqrt(np.nanmean(((y - np.nanmean(y)) - (x - np.nanmean(x)))**2))
    #     bias = np.nanmean(y) - np.nanmean(x)
    #     cci_fit_stats = [r_value**2, spear_r, crmse, bias]
    #     fit_stats_df.loc[('MCD19', 'CCI'), :] = [r_value**2, bias, crmse, spear_r, spear_p]

    #     print(f"MCD19 vs PhotoSpec CCI Fit parameters (interp): R^2={r_value**2:.3f}, bias={bias:.3f}, CRMSE={crmse:.3f}, Spearman r={spear_r:.3f}, p-value={spear_p:.3e}")
    # else:
    #     print("Not enough valid data for regression (interp).")



    # # Only compare where both are valid (not nan)
    # valid_mask = (~np.isnan(MCD19_QC_pt['NDVI'].values)) & (~np.isnan(photospec_ndvi_interp.values))
    # # Compute R^2 for the scatter of interpolated Photospec NDVI vs filtered MCD19 NDVI
    # if np.sum(valid_mask) > 1:
    #     slope, intercept, r_value, p_value, std_err = linregress(
    #         photospec_ndvi_interp.values[valid_mask],
    #         MCD19_QC_pt['NDVI'].values[valid_mask]
    #     )
    #     # Spearman rank correlation
    #     spear_r, spear_p = spearmanr(photospec_ndvi_interp.values[valid_mask], MCD19_QC_pt['NDVI'].values[valid_mask])

    #     # Centered (unbiased) RMSE (CRMSE): removes mean bias
    #     x = photospec_ndvi_interp.values[valid_mask]
    #     y = MCD19_QC_pt['NDVI'].values[valid_mask]
    #     crmse = np.sqrt(np.nanmean(((y - np.nanmean(y)) - (x - np.nanmean(x)))**2))
    #     bias = np.nanmean(y) - np.nanmean(x)
    #     ndvi_fit_stats = [r_value**2, spear_r, crmse, bias]
    #     fit_stats_df.loc[('MCD19', 'CCI'), :] = [r_value**2, bias, crmse, spear_r, spear_p]

    #     print(f"MCD19 vs PhotoSpec NDVI Fit parameters (interp): R^2={r_value**2:.3f}, bias={bias:.3f}, CRMSE={crmse:.3f}, Spearman r={spear_r:.3f}, p-value={spear_p:.3e}")
    # else:
    #     print("Not enough valid data for regression (interp).")



    # ######################################################################
    # ### For comparison, check PhotoSpec vs. MOD09 fit params
    # ######################################################################

    # MOD09_interp = MOD09_pt.sel(time=slice(photospec_cci_interp.index[0], Photospec_times_filtered.max()+pd.Timedelta(days=16)))

    # # Check that the interpolated MOD09 series is aligned with photospec_cci_interp.
    # mod09_interp_times = pd.DatetimeIndex(pd.to_datetime(MOD09_interp.time.values))
    # photospec_cci_interp_times = pd.DatetimeIndex(pd.to_datetime(photospec_cci_interp.index))
    # if not mod09_interp_times.equals(photospec_cci_interp_times):
    #     raise ValueError("MOD09_interp and photospec_cci_interp timestamps do not match")
    # print(f"MOD09_interp timestamps match photospec_cci_interp ({len(MOD09_interp.time)} timestamps)")



    # # Only compare where both are valid (not nan)
    # valid_mask = (~np.isnan(MOD09_interp['CCI'].values)) & (~np.isnan(photospec_cci_interp.values))
    # # Compute R^2 for the scatter of interpolated Photospec NDVI vs filtered MCD19 NDVI
    # if np.sum(valid_mask) > 1:
    #     slope, intercept, r_value, p_value, std_err = linregress(
    #         photospec_cci_interp.values[valid_mask],
    #         MOD09_interp['CCI'].values[valid_mask]
    #     )
    #     # Spearman rank correlation
    #     spear_r, spear_p = spearmanr(photospec_cci_interp.values[valid_mask], MOD09_interp['CCI'].values[valid_mask])

    #     # Centered (unbiased) RMSE (CRMSE): removes mean bias
    #     x = photospec_cci_interp.values[valid_mask]
    #     y = MOD09_interp['CCI'].values[valid_mask]
    #     crmse = np.sqrt(np.nanmean(((y - np.nanmean(y)) - (x - np.nanmean(x)))**2))
    #     bias = np.nanmean(y) - np.nanmean(x)
    #     mod09_cci_fit_stats = [r_value**2, spear_r, crmse, bias]
    #     fit_stats_df.loc[('MOD09', 'CCI'), :] = [r_value**2, bias, crmse, spear_r, spear_p]

    #     print(f"MOD09 vs PhotoSpec CCI Fit parameters (interp): R^2={r_value**2:.3f}, bias={bias:.3f}, CRMSE={crmse:.3f}, Spearman r={spear_r:.3f}, p-value={spear_p:.3e}")
    # else:
    #     print("Not enough valid data for regression (interp).")





    # # Only compare where both are valid (not nan)
    # valid_mask = (~np.isnan(MOD09_interp['NDVI'].values)) & (~np.isnan(photospec_ndvi_interp.values))
    # # Compute R^2 for the scatter of interpolated Photospec NDVI vs filtered MCD19 NDVI
    # if np.sum(valid_mask) > 1:
    #     slope, intercept, r_value, p_value, std_err = linregress(
    #         photospec_ndvi_interp.values[valid_mask],
    #         MOD09_interp['NDVI'].values[valid_mask]
    #     )
    #     # Spearman rank correlation
    #     spear_r, spear_p = spearmanr(photospec_ndvi_interp.values[valid_mask], MOD09_interp['NDVI'].values[valid_mask])

    #     # Centered (unbiased) RMSE (CRMSE): removes mean bias
    #     x = photospec_ndvi_interp.values[valid_mask]
    #     y = MOD09_interp['NDVI'].values[valid_mask]
    #     crmse = np.sqrt(np.nanmean(((y - np.nanmean(y)) - (x - np.nanmean(x)))**2))
    #     bias = np.nanmean(y) - np.nanmean(x)
    #     ndvi_fit_stats = [r_value**2, spear_r, crmse, bias]
    #     fit_stats_df.loc[('MOD09', 'NDVI'), :] = [r_value**2, bias, crmse, spear_r, spear_p]

    #     print(f"MOD09 vs PhotoSpec NDVI Fit parameters (interp): R^2={r_value**2:.3f}, bias={bias:.3f}, CRMSE={crmse:.3f}, Spearman r={spear_r:.3f}, p-value={spear_p:.3e}")
    # else:
    #     print("Not enough valid data for regression (interp).")




    # ######################################################################
    # ### For comparison, check PhotoSpec vs. MOD13 fit params
    # ######################################################################

    # MOD13_interp = MOD13_pt.sel(time=slice(photospec_cci_interp.index[0], Photospec_times_filtered.max()+pd.Timedelta(days=16)))

    # # Check that the interpolated MOD13 series is aligned with photospec_cci_interp.
    # MOD13_interp_times = pd.DatetimeIndex(pd.to_datetime(MOD13_interp.time.values))
    # if not MOD13_interp_times.equals(photospec_cci_interp_times):
    #     raise ValueError("MOD13_interp and photospec_cci_interp timestamps do not match")
    # print(f"MOD13_interp timestamps match photospec_cci_interp ({len(MOD13_interp.time)} timestamps)")


    # # Only compare where both are valid (not nan)
    # valid_mask = (~np.isnan(MOD13_interp['NDVI'].values)) & (~np.isnan(photospec_ndvi_interp.values))
    # # Compute R^2 for the scatter of interpolated Photospec NDVI vs filtered MCD19 NDVI
    # if np.sum(valid_mask) > 1:
    #     slope, intercept, r_value, p_value, std_err = linregress(
    #         photospec_ndvi_interp.values[valid_mask],
    #         MOD13_interp['NDVI'].values[valid_mask]
    #     )
    #     # Spearman rank correlation
    #     spear_r, spear_p = spearmanr(photospec_ndvi_interp.values[valid_mask], MOD13_interp['NDVI'].values[valid_mask])

    #     # Centered (unbiased) RMSE (CRMSE): removes mean bias
    #     x = photospec_ndvi_interp.values[valid_mask]
    #     y = MOD13_interp['NDVI'].values[valid_mask]
    #     crmse = np.sqrt(np.nanmean(((y - np.nanmean(y)) - (x - np.nanmean(x)))**2))
    #     bias = np.nanmean(y) - np.nanmean(x)
    #     ndvi_fit_stats = [r_value**2, spear_r, crmse, bias]
    #     fit_stats_df.loc[('MOD13', 'NDVI'), :] = [r_value**2, bias, crmse, spear_r, spear_p]

    #     print(f"MOD13 vs PhotoSpec NDVI Fit parameters (interp): R^2={r_value**2:.3f}, bias={bias:.3f}, CRMSE={crmse:.3f}, Spearman r={spear_r:.3f}, p-value={spear_p:.3e}")
    # else:
    #     print("Not enough valid data for regression (interp).")




    # ######################################################################
    # ### For comparison, check PhotoSpec vs. MOD13 fit params
    # ######################################################################


    # # First, Interpolate Photospec CCI to match MCD19_QC_pt.time
    # photospec_cci_interp_FOR_GCOM_COMPARE = pd.Series(
    #     np.interp(
    #         pd.to_datetime(GCOM_pt.time.values).astype(np.int64),
    #         Photospec_times_filtered.astype(np.int64),
    #         Photospec_df_filtered['CCI'].values
    #     ),
    #     index=GCOM_pt.time.values
    # )


    # # First, Interpolate Photospec CCI to match MCD19_QC_pt.time
    # photospec_ndvi_interp_FOR_GCOM_COMPARE = pd.Series(
    #     np.interp(
    #         pd.to_datetime(GCOM_pt.time.values).astype(np.int64),
    #         Photospec_times_filtered.astype(np.int64),
    #         Photospec_df_filtered['NDVI'].values
    #     ),
    #     index=GCOM_pt.time.values
    # )


    # # Set to nan before first and after last valid Photospec time
    # first_time = Photospec_times_filtered.iloc[0]
    # last_time = Photospec_times_filtered.iloc[-1]
    # interp_times = pd.to_datetime(GCOM_pt.time.values)

    # photospec_cci_interp_FOR_GCOM_COMPARE[(interp_times < first_time) | (interp_times > last_time)] = np.nan
    # photospec_ndvi_interp_FOR_GCOM_COMPARE[(interp_times < first_time) | (interp_times > last_time)] = np.nan

    # # if not GCOM_pt.time.equals(photospec_cci_interp_times_FOR_GCOM_COMPARE):
    # #     raise ValueError("GCOM_interp and photospec_cci_interp timestamps do not match")
    # # print(f"GCOM_interp timestamps match photospec_cci_interp ({len(GCOM_interp.time)} timestamps)")



    # # Only compare where both are valid (not nan)
    # valid_mask = (~np.isnan(GCOM_pt['CCI'].values)) & (~np.isnan(photospec_cci_interp_FOR_GCOM_COMPARE.values))
    # # Compute R^2 for the scatter of interpolated Photospec CCI vs filtered MCD19 CCI
    # if np.sum(valid_mask) > 1:
    #     slope, intercept, r_value, p_value, std_err = linregress(
    #         photospec_cci_interp_FOR_GCOM_COMPARE.values[valid_mask],
    #         GCOM_pt['CCI'].values[valid_mask]
    #     )
    #     # Spearman rank correlation
    #     spear_r, spear_p = spearmanr(photospec_cci_interp_FOR_GCOM_COMPARE.values[valid_mask], GCOM_pt['CCI'].values[valid_mask])

    #     # Centered (unbiased) RMSE (CRMSE): removes mean bias
    #     x = photospec_cci_interp_FOR_GCOM_COMPARE.values[valid_mask]
    #     y = GCOM_pt['CCI'].values[valid_mask]
    #     crmse = np.sqrt(np.nanmean(((y - np.nanmean(y)) - (x - np.nanmean(x)))**2))
    #     bias = np.nanmean(y) - np.nanmean(x)
    #     cci_fit_stats = [r_value**2, spear_r, crmse, bias]
    #     fit_stats_df.loc[('GCOM', 'CCI'), :] = [r_value**2, bias, crmse, spear_r, spear_p]

    #     print(f"GCOM vs PhotoSpec CCI Fit parameters (interp): R^2={r_value**2:.3f}, bias={bias:.3f}, CRMSE={crmse:.3f}, Spearman r={spear_r:.3f}, p-value={spear_p:.3e}")
    # else:
    #     print("Not enough valid data for regression (interp).")


    # # Only compare where both are valid (not nan)
    # valid_mask = (~np.isnan(GCOM_pt['NDVI'].values)) & (~np.isnan(photospec_ndvi_interp_FOR_GCOM_COMPARE.values))
    # # Compute R^2 for the scatter of interpolated Photospec NDVI vs filtered MCD19 NDVI
    # if np.sum(valid_mask) > 1:
    #     slope, intercept, r_value, p_value, std_err = linregress(
    #         photospec_ndvi_interp_FOR_GCOM_COMPARE.values[valid_mask],
    #         GCOM_pt['NDVI'].values[valid_mask]
    #     )
    #     # Spearman rank correlation
    #     spear_r, spear_p = spearmanr(photospec_ndvi_interp_FOR_GCOM_COMPARE.values[valid_mask], GCOM_pt['NDVI'].values[valid_mask])

    #     # Centered (unbiased) RMSE (CRMSE): removes mean bias
    #     x = photospec_ndvi_interp_FOR_GCOM_COMPARE.values[valid_mask]
    #     y = GCOM_pt['NDVI'].values[valid_mask]
    #     crmse = np.sqrt(np.nanmean(((y - np.nanmean(y)) - (x - np.nanmean(x)))**2))
    #     bias = np.nanmean(y) - np.nanmean(x)
    #     ndvi_fit_stats = [r_value**2, spear_r, crmse, bias]
    #     fit_stats_df.loc[('GCOM', 'NDVI'), :] = [r_value**2, bias, crmse, spear_r, spear_p]

    #     print(f"GCOM vs PhotoSpec NDVI Fit parameters (interp): R^2={r_value**2:.3f}, bias={bias:.3f}, CRMSE={crmse:.3f}, Spearman r={spear_r:.3f}, p-value={spear_p:.3e}")
    # else:
    #     print("Not enough valid data for regression (interp).")




    # ######################################################################
    # ### For comparison, check PhotoSpec vs. MCD43 fit params
    # ######################################################################

    # MCD43_interp = MCD43_pt.interp(time=photospec_ndvi_interp.index)

    # # Check that the interpolated MCD43 series is aligned with photospec_cci_interp.
    # MCD43_interp_times = pd.DatetimeIndex(pd.to_datetime(MCD43_interp.time.values))
    # if not MCD43_interp_times.equals(photospec_cci_interp_times):
    #     raise ValueError("MCD43_interp and photospec_cci_interp timestamps do not match")
    # print(f"MCD43_interp timestamps match photospec_cci_interp ({len(MCD43_interp.time)} timestamps)")



    # # Only compare where both are valid (not nan)
    # valid_mask = (~np.isnan(MCD43_interp['NDVI'].values)) & (~np.isnan(photospec_ndvi_interp.values))
    # # Compute R^2 for the scatter of interpolated Photospec NDVI vs filtered MCD19 NDVI
    # if np.sum(valid_mask) > 1:
    #     slope, intercept, r_value, p_value, std_err = linregress(
    #         photospec_ndvi_interp.values[valid_mask],
    #         MCD43_interp['NDVI'].values[valid_mask]
    #     )
    #     # Spearman rank correlation
    #     spear_r, spear_p = spearmanr(photospec_ndvi_interp.values[valid_mask], MCD43_interp['NDVI'].values[valid_mask])

    #     # Centered (unbiased) RMSE (CRMSE): removes mean bias
    #     x = photospec_ndvi_interp.values[valid_mask]
    #     y = MCD43_interp['NDVI'].values[valid_mask]
    #     crmse = np.sqrt(np.nanmean(((y - np.nanmean(y)) - (x - np.nanmean(x)))**2))
    #     bias = np.nanmean(y) - np.nanmean(x)
    #     ndvi_fit_stats = [r_value**2, spear_r, crmse, bias]
    #     fit_stats_df.loc[('MCD43', 'NDVI'), :] = [r_value**2, bias, crmse, spear_r, spear_p]

    #     print(f"MCD43 vs PhotoSpec NDVI Fit parameters (interp): R^2={r_value**2:.3f}, bias={bias:.3f}, CRMSE={crmse:.3f}, Spearman r={spear_r:.3f}, p-value={spear_p:.3e}")
    # else:
    #     print("Not enough valid data for regression (interp).")

    # fit_stats_outfile = os.path.join(out_stats_dir, f'PhotoSpec_v_MCD19_etc_fit_stats_{site_name}.csv')
    # fit_stats_df.to_csv(fit_stats_outfile)


    #%%
    ### From Claude:

    #~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#
    ### Calculate fit params (generalized time alignment)
    ### Replaces everything from "### Calculate fit params" to the end of the script.
    #~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#

    # Longest gap (days) between consecutive valid PhotoSpec observations that interpolation
    # is allowed to bridge. Product timestamps falling inside a longer gap are set to NaN.
    # 16 matches the rule in the 01_QC_compare and 02_gapfill_compare scripts. None = no limit.
    MAX_GAP_DAYS = 16

    # Minimum number of paired timesteps required to compute stats
    MIN_N = 3


    def to_ns_int(times):
        """Convert any datetime-like array (datetime64 of any unit, pandas, cftime) to int64 ns."""
        times = np.asarray(times)
        if times.size and isinstance(times.flat[0], cftime.datetime):
            times = xr.CFTimeIndex(times).to_datetimeindex().values
        return pd.to_datetime(times).values.astype('datetime64[ns]').astype(np.int64)


    def interp_photospec_to_times(target_times, ps_times, ps_vals, max_gap_days=None):
        """
        Linearly interpolate PhotoSpec values onto target_times.
        - NaN PhotoSpec values are dropped before interpolating (np.interp does not skip NaNs)
        - Targets outside the PhotoSpec time range are set to NaN
        - Optionally, targets that fall strictly inside a gap between consecutive valid PhotoSpec
          observations longer than max_gap_days are set to NaN
        """
        xt = to_ns_int(target_times)
        x = to_ns_int(ps_times)
        v = np.asarray(ps_vals, dtype=float)

        ok = ~np.isnan(v)
        x, v = x[ok], v[ok]
        order = np.argsort(x)
        x, v = x[order], v[order]
        x, first_idx = np.unique(x, return_index=True)  # drop duplicate timestamps
        v = v[first_idx]

        out = np.full(xt.shape, np.nan)
        if x.size < 2:
            return out

        out = np.interp(xt, x, v)
        out[(xt < x[0]) | (xt > x[-1])] = np.nan

        if max_gap_days is not None:
            long_gap = np.diff(x) > pd.Timedelta(days=max_gap_days).value  # gap after obs i
            left = np.searchsorted(x, xt, side='right') - 1                # left neighbor index
            in_interval = (left >= 0) & (left < x.size - 1)
            exact_hit = np.zeros(xt.shape, dtype=bool)
            exact_hit[left >= 0] = x[left[left >= 0]] == xt[left >= 0]
            bad = np.zeros(xt.shape, dtype=bool)
            bad[in_interval] = long_gap[left[in_interval]]
            out[bad & ~exact_hit] = np.nan

        return out


    def fit_stats(x, y):
        """x = PhotoSpec (reference), y = satellite product."""
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        spear_r, spear_p = spearmanr(x, y)
        crmse = np.sqrt(np.mean(((y - y.mean()) - (x - x.mean()))**2))
        bias = y.mean() - x.mean()
        return {'R2': r_value**2, 'bias': bias, 'CRMSE': crmse,
                'Spearman_r': spear_r, 'Spearman_pval': spear_p, 'N': len(x)}


    def compare_product(ds, var, ps_times, ps_df, max_gap_days=None, min_n=3):
        """
        Pair a point time series from `ds` with PhotoSpec interpolated to the product's
        own timestamps, keep only timesteps where both are valid, and compute stats.
        Returns (stats dict or None, paired DataFrame or None).
        """
        if var not in ds or var not in ps_df.columns:
            return None, None

        y = np.asarray(ds[var].values, dtype=float).squeeze()
        if y.ndim != 1:
            raise ValueError(f"{var} has shape {ds[var].shape} after squeeze; expected a 1-D time series")

        t = ds['time'].values
        ps_on_t = interp_photospec_to_times(t, ps_times, ps_df[var].values, max_gap_days)

        pair = pd.DataFrame({'photospec': ps_on_t, 'product': y},
                            index=pd.DatetimeIndex(to_ns_int(t).astype('datetime64[ns]'), name='time'))
        pair = pair.dropna()

        if len(pair) < min_n:
            return None, pair
        return fit_stats(pair['photospec'].values, pair['product'].values), pair


    # ---------------------------------------------------------------------
    # Run all product/variable comparisons
    # ---------------------------------------------------------------------
    products = {
        'MCD19': MCD19_QC_pt,
        'MCD43': MCD43_pt.interp(time=MCD19_QC_pt.time),
        'MOD09': MOD09_pt,
        'MOD13': MOD13_pt,
        'GCOM': GCOM_pt,
    }
    elements = ['CCI', 'NDVI']
    stat_cols = ['R2', 'bias', 'CRMSE', 'Spearman_r', 'Spearman_pval', 'N']

    rows = []
    paired = {}  # (product, element) -> paired DataFrame, for sanity plots
    for prod, ds in products.items():
        for var in elements:
            stats, pair = compare_product(ds, var, Photospec_times_filtered, Photospec_df_filtered,
                                        max_gap_days=MAX_GAP_DAYS, min_n=MIN_N)
            paired[(prod, var)] = pair
            if stats is None:
                n = 0 if pair is None else len(pair)
                reason = 'variable not present' if pair is None else f'only {n} paired timesteps'
                print(f"{prod} {var}: skipped ({reason})")
                stats = {c: np.nan for c in stat_cols}
                stats['N'] = n
            else:
                print(f"{prod} vs PhotoSpec {var}: R^2={stats['R2']:.3f}, bias={stats['bias']:.3f}, "
                    f"CRMSE={stats['CRMSE']:.3f}, Spearman r={stats['Spearman_r']:.3f}, "
                    f"p={stats['Spearman_pval']:.3e}, N={stats['N']}")
            rows.append({'QC': prod, 'element': var, **stats})

    fit_stats_df = pd.DataFrame(rows).set_index(['QC', 'element'])[stat_cols]
    print(fit_stats_df.round(3))


    # ---------------------------------------------------------------------
    # Optional sanity check: paired values by day of year
    # ---------------------------------------------------------------------
    PLOT_SANITY = False
    if PLOT_SANITY:
        for (prod, var), pair in paired.items():
            if pair is None or len(pair) == 0:
                continue
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.scatter(Photospec_times_filtered.dt.dayofyear, Photospec_df_filtered[var],
                    color='gray', s=14, alpha=0.5, label=f'PhotoSpec daily, {overpass_window[0]}-{overpass_window[1]}')
            ax.scatter(pair.index.dayofyear, pair['photospec'], color='black', s=28,
                    marker='^', label='PhotoSpec interp')
            ax.scatter(pair.index.dayofyear, pair['product'], color='red', s=28,
                    marker='s', label=prod)
            ax.set_ylabel(var)
            ax.set_xticks(month_starts)
            ax.set_xticklabels(month_labels)
            ax.set_title(f'{site_name}: {prod} {var} (N={len(pair)})')
            ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
            plt.tight_layout()
            plt.show()


    fit_stats_outfile = os.path.join(out_stats_dir, f'PhotoSpec_v_MCD19_etc_fit_stats_{site_name}.csv')
    fit_stats_df.to_csv(fit_stats_outfile)

if __name__ == "__main__":
    main()