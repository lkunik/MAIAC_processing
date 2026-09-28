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
    ### QC ANALYSIS DELIVERABLE 1 - NDVI time series (with MCD19 ± sigma)
    ############################################

    mcd19_ndvi = MCD19_QC_pt['NDVI'].values
    mcd19_ndvi_sig = MCD19_QC_pt['NDVI_sigma'].values
    mcd19_ndvi_lo = mcd19_ndvi - mcd19_ndvi_sig
    mcd19_ndvi_hi = mcd19_ndvi + mcd19_ndvi_sig

    ndvi_arrays = [
        mcd19_ndvi_lo,
        mcd19_ndvi_hi,
        Photospec_df_filtered['NDVI'].values
    ]
    mcd19_ndvi_min = np.nanmin(np.concatenate(ndvi_arrays))
    mcd19_ndvi_max = np.nanmax(np.concatenate(ndvi_arrays))
    mcd19_ndvi_range = mcd19_ndvi_max - mcd19_ndvi_min

    fig, ax = plt.subplots(figsize=(7, 3))

    for start, end in winter_periods:
        ax.axvspan(start, end, color='lightblue', alpha=0.5, zorder=0)
    ax.scatter(Photospec_times_filtered, Photospec_df_filtered['NDVI'], color='black', s=10, alpha=0.7)

    ax.scatter(MOD13_times, MOD13_pt['NDVI'].values, color=MOD13_color, s=25, alpha=1, marker='s', edgecolors='black', linewidths=0.5)
    ax.scatter(MCD43_times, MCD43_pt['NDVI'].values, color=MCD43_color, s=25, alpha=1, marker='s', edgecolors='black', linewidths=0.5)

    ax.scatter(MOD09_times, MOD09_pt['NDVI'].values, color=MOD09_color, s=45, alpha=1, marker='^', edgecolors='black', linewidths=0.5, zorder=2)
    ax.scatter(GCOM_times, GCOM_pt['NDVI'].values, color=GCOM_color, s=45, alpha=1, marker='^', edgecolors='black', linewidths=0.5, zorder=2)

    # MCD19 uncertainty band (± NDVI_sigma)
    ax.fill_between(MCD19_QC_times, mcd19_ndvi_lo, mcd19_ndvi_hi,
                    color=MCD19_QC_color, alpha=0.25, linewidth=0, zorder=1.5)
    ax.plot(MCD19_QC_times, mcd19_ndvi, color=MCD19_QC_color, marker='o', markersize=8, linewidth=3.5,
            alpha=1, markeredgecolor='black', markeredgewidth=0.5, zorder=3)

    ax.set_ylim(mcd19_ndvi_min - (mcd19_ndvi_range*0.15), mcd19_ndvi_max + (mcd19_ndvi_range*0.05))
    ax.yaxis.set_major_formatter(FFmt(lambda y, _: f'{y:.1f}'))
    # ax.yaxis.set_major_locator(MultipleLocator(0.3))

    ax.set_xlim(Photospec_times_filtered.min() - pd.Timedelta(days=10), Photospec_times_filtered.max() + pd.Timedelta(days=10))
    ax.xaxis.set_major_locator(MonthLocator(bymonth=[7, 10], interval=1))
    ax.xaxis.set_major_locator(MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(DateFormatter('%b\n%Y'))
    ax.set_title(f'e)  {plot_title}', loc='center')
    ax.set_ylabel('NDVI')

    plt.tight_layout()
    ndvi_panel_file = os.path.join(panel_dir, f'{site_name}_NDVI_sigma_panel.png')
    plt.savefig(ndvi_panel_file, dpi=300, bbox_inches='tight')


    ############################################
    ### QC ANALYSIS DELIVERABLE - CCI time series (with MCD19 ± sigma)
    ############################################

    mcd19_cci = MCD19_QC_pt['CCI'].values
    mcd19_cci_sig = MCD19_QC_pt['CCI_sigma'].values
    mcd19_cci_lo = mcd19_cci - mcd19_cci_sig
    mcd19_cci_hi = mcd19_cci + mcd19_cci_sig

    cci_arrays = [
        mcd19_cci_lo,
        mcd19_cci_hi,
        Photospec_df_filtered['CCI'].values
    ]
    mcd19_cci_min = np.nanmin(np.concatenate(cci_arrays))
    mcd19_cci_max = np.nanmax(np.concatenate(cci_arrays))
    mcd19_cci_range = mcd19_cci_max - mcd19_cci_min

    fig, ax = plt.subplots(figsize=(7, 3))

    for start, end in winter_periods:
        ax.axvspan(start, end, color='lightblue', alpha=0.5, zorder=0)
    ax.scatter(Photospec_times_filtered, Photospec_df_filtered['CCI'], color='black', s=10, alpha=0.7)

    ax.scatter(MOD09_times, MOD09_pt['CCI'].values, color=MOD09_color, s=45, alpha=1, marker='^', edgecolors='black', linewidths=0.5, zorder=2)
    ax.scatter(GCOM_times, GCOM_pt['CCI'].values, color=GCOM_color, s=45, alpha=1, marker='^', edgecolors='black', linewidths=0.5, zorder=2)

    # MCD19 uncertainty band (± CCI_sigma)
    ax.fill_between(MCD19_QC_times, mcd19_cci_lo, mcd19_cci_hi,
                    color=MCD19_QC_color, alpha=0.25, linewidth=0, zorder=1.5)
    ax.plot(MCD19_QC_times, mcd19_cci, color=MCD19_QC_color, marker='o', markersize=8, linewidth=3.5,
            alpha=1, markeredgecolor='black', markeredgewidth=0.5, zorder=3)

    ax.set_ylim(mcd19_cci_min - (mcd19_cci_range*0.05), mcd19_cci_max + (mcd19_cci_range*0.05))
    ax.yaxis.set_major_formatter(FFmt(lambda y, _: f'{y:.1f}'))
    # ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_xlim(Photospec_times_filtered.min() - pd.Timedelta(days=10), Photospec_times_filtered.max() + pd.Timedelta(days=10))
    ax.xaxis.set_major_locator(MonthLocator(bymonth=[7, 10], interval=1))
    ax.xaxis.set_major_locator(MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(DateFormatter('%b\n%Y'))
    ax.set_title(f'f)  {plot_title}', loc='center')
    ax.set_ylabel('CCI')

    plt.tight_layout()
    cci_panel_file = os.path.join(panel_dir, f'{site_name}_CCI_sigma_panel.png')
    plt.savefig(cci_panel_file, dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()