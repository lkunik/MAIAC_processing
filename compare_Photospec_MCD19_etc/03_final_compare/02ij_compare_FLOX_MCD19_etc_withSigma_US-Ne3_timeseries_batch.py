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
plot_title = 'US-Ne3'
MODIS_tile = 'h10v04'

QC_descr = "CloudFree_LowAOD_ClearAdj"

MCD19_legend_descr = "CV-MVC" # gets added in parentheses after "MCD19"


dat_pt_basedir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/data/QC_PhotoSpec_sites_backup/'
dat_pt_basedir_QC = os.path.join(dat_pt_basedir, f'CV-MVC/lm-fill/QCfilt_{QC_descr}')

MCD19_QC_file = os.path.join(dat_pt_basedir_QC, f'MCD19_QCfilt_{QC_descr}_{site_name}.nc')

FloX_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/data/FloX/US-Ne3/'
FloX_2018_file = os.path.join(FloX_dir, f'DFlox_SIF_VIs_2018.csv')
FloX_2019_file = os.path.join(FloX_dir, f'DFlox_SIF_VIs_2019.csv')

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

    MCD19_QC_pt = xr.open_dataset(MCD19_QC_file)

    MOD13_pt = xr.open_dataset(MOD13_file)
    MCD43_pt = xr.open_dataset(MCD43_file)
    GCOM_pt = xr.open_dataset(GCOM_file)
    MOD09_pt = xr.open_dataset(MOD09_file)

    FloX_df_2018 = pd.read_csv(FloX_2018_file)


    print("FloX_df_2018 columns:", FloX_df_2018.columns.tolist())
    print(f'FloX_df_2018 rows (before filtering for valid datetimes): {len(FloX_df_2018)}')
    FloX_df_2018 = FloX_df_2018[FloX_df_2018['DoY'].between(1, 366)]
    FloX_df_2018['DateTime'] = pd.to_datetime(FloX_df_2018['DateTime'])
    FloX_df_2018 = FloX_df_2018[FloX_df_2018['DateTime'].between('2018-01-01', '2018-12-31')]
    FloX_df_2018 = FloX_df_2018[
        FloX_df_2018['DateTime'].dt.time.between(pd.Timestamp('13:00').time(), pd.Timestamp('14:00').time())
    ]

    FloX_df_2018 = FloX_df_2018[FloX_df_2018['NDVI'].between(0, 1)]
    FloX_df_2018 = remove_running_window_outliers(FloX_df_2018)




    print(f'FloX_df_2018 rows (after filtering for valid datetimes): {len(FloX_df_2018)}\n\n\n')

    FloX_df_2019 = pd.read_csv(FloX_2019_file)
    print(f'FloX_df_2019 rows (before filtering for valid datetimes): {len(FloX_df_2019)}')
    FloX_df_2019['DoY'] = FloX_df_2019['DoY'] - 365
    FloX_df_2019 = FloX_df_2019[FloX_df_2019['DoY'].between(1, 366)]
    FloX_df_2019['DateTime'] = pd.to_datetime(FloX_df_2019['DateTime'])
    FloX_df_2019 = FloX_df_2019[FloX_df_2019['DateTime'].between('2019-01-01', '2019-12-31')]
    FloX_df_2019 = FloX_df_2019[
        FloX_df_2019['DateTime'].dt.time.between(pd.Timestamp('13:00').time(), pd.Timestamp('14:00').time())
    ]
    FloX_df_2019 = FloX_df_2019[FloX_df_2019['NDVI'].between(0, 1)]
    FloX_df_2019 = remove_running_window_outliers(FloX_df_2019)
    print(f'FloX_df_2019 rows (after filtering for valid datetimes): {len(FloX_df_2019)}')


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

    # MCD19_QC_pt = MCD19_QC_pt.sel(time=slice(FloX_times.iloc[0], FloX_times.iloc[-1]))
    MCD19_QC_pt = MCD19_QC_pt.sel(time=slice(
        FloX_times_filtered.iloc[0] - pd.Timedelta(days=16),
        FloX_times_filtered.iloc[-1] + pd.Timedelta(days=16),
    ))

    MCD19_QC_times = np.array([np.datetime64(t) for t in MCD19_QC_pt['time'].values])

    MOD13_times = np.array([np.datetime64(t) for t in MOD13_pt['time'].values])
    MCD43_times = np.array([np.datetime64(t) for t in MCD43_pt['time'].values])
    GCOM_times = np.array([np.datetime64(t) for t in GCOM_pt['time'].values])
    MOD09_times = np.array([np.datetime64(t) for t in MOD09_pt['time'].values])

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
        (pd.Timestamp('2017-12-24'), pd.Timestamp('2018-02-25')),
        (pd.Timestamp('2018-11-09'), pd.Timestamp('2019-03-15')),
        (pd.Timestamp('2019-12-15'), pd.Timestamp('2020-02-08')),
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
        FloX_df_filtered['NDVI']
    ]
    mcd19_ndvi_min = np.nanmin(np.concatenate(ndvi_arrays))
    mcd19_ndvi_max = np.nanmax(np.concatenate(ndvi_arrays))
    mcd19_ndvi_range = mcd19_ndvi_max - mcd19_ndvi_min


    #%%
    fig, ax = plt.subplots(figsize=(7, 3))

    for start, end in winter_periods:
        ax.axvspan(start, end, color='lightblue', alpha=0.5, zorder=0)
    ax.scatter(FloX_times_filtered, FloX_df_filtered['NDVI'], color='black', s=10, alpha=0.7)


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


    ax.set_xlim(FloX_times_filtered.min() - pd.Timedelta(days=10), FloX_times_filtered.max() + pd.Timedelta(days=10))
    ax.xaxis.set_major_locator(MonthLocator(bymonth=[7, 10], interval=1))
    ax.xaxis.set_major_locator(MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(DateFormatter('%b\n%Y'))
    # ax.set_title('i)', loc='left')
    ax.set_title(f'i)  {plot_title}', loc='center')
    ax.set_ylabel('NDVI')

    plt.tight_layout()
    ndvi_panel_file = os.path.join(panel_dir, f'{site_name}_NDVI_sigma_panel.png')
    plt.savefig(ndvi_panel_file, dpi=300, bbox_inches='tight')


    #%%

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
        FloX_df_filtered['CCI']
    ]
    mcd19_cci_min = np.nanmin(np.concatenate(cci_arrays))
    mcd19_cci_max = np.nanmax(np.concatenate(cci_arrays))
    mcd19_cci_range = mcd19_cci_max - mcd19_cci_min

    fig, ax = plt.subplots(figsize=(7, 3))

    for start, end in winter_periods:
        ax.axvspan(start, end, color='lightblue', alpha=0.5, zorder=0)
    ax.scatter(FloX_times_filtered, FloX_df_filtered['CCI'], color='black', s=10, alpha=0.7)

    ax.scatter(MOD09_times, MOD09_pt['CCI'].values, color=MOD09_color, s=45, alpha=1, marker='^', edgecolors='black', linewidths=0.5, zorder=2)
    ax.scatter(GCOM_times, GCOM_pt['CCI'].values, color=GCOM_color, s=45, alpha=1, marker='^', edgecolors='black', linewidths=0.5, zorder=2)

    # MCD19 uncertainty band (± CCI_sigma)
    ax.fill_between(MCD19_QC_times, mcd19_cci_lo, mcd19_cci_hi,
                    color=MCD19_QC_color, alpha=0.25, linewidth=0, zorder=1.5)
    ax.plot(MCD19_QC_times, mcd19_cci, color=MCD19_QC_color, marker='o', markersize=8, linewidth=3.5,
            alpha=1, markeredgecolor='black', markeredgewidth=0.5, zorder=3)

    ax.set_ylim(mcd19_cci_min - (mcd19_cci_range*0.05), mcd19_cci_max + (mcd19_cci_range*0.05))
    ax.yaxis.set_major_formatter(FFmt(lambda y, _: f'{y:.1f}'))
    # ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.set_xlim(FloX_times_filtered.min() - pd.Timedelta(days=10), FloX_times_filtered.max() + pd.Timedelta(days=10))
    ax.xaxis.set_major_locator(MonthLocator(bymonth=[7, 10], interval=1))
    ax.xaxis.set_major_locator(MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(DateFormatter('%b\n%Y'))
    # ax.set_title('j)', loc='left')
    ax.set_title(f'j)  {plot_title}', loc='center')
    ax.set_ylabel('CCI')

    plt.tight_layout()
    cci_panel_file = os.path.join(panel_dir, f'{site_name}_CCI_sigma_panel.png')
    plt.savefig(cci_panel_file, dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()