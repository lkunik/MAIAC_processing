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
from matplotlib.ticker import MultipleLocator, MaxNLocator
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


QC_descr = "CloudFree_LowAOD_ClearAdj"

out_stats_descr = "GF_Compare"

dat_pt_basedir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/data/QC_PhotoSpec_sites_backup/'
dat_pt_basedir_prefill = os.path.join(dat_pt_basedir, f'CV-MVC/pre-fill/QCfilt_{QC_descr}')
dat_pt_basedir_prefill_OutlierFilter = os.path.join(dat_pt_basedir, f'CV-MVC/pre-fill_OutlierFilter/QCfilt_{QC_descr}')
dat_pt_basedir_hist_avg = os.path.join(dat_pt_basedir, f'CV-MVC/hist_avg/QCfilt_{QC_descr}')
dat_pt_basedir_hist_avg_OutlierFilter = os.path.join(dat_pt_basedir, f'CV-MVC/hist_avg_OutlierFilter/QCfilt_{QC_descr}')
dat_pt_basedir_lmfill = os.path.join(dat_pt_basedir, f'CV-MVC/lm-fill/QCfilt_{QC_descr}')
dat_pt_basedir_basicfill = os.path.join(dat_pt_basedir, f'CV-MVC/basic-fill/QCfilt_{QC_descr}')

Photospec_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/data/PhotoSpec'
Photospec_file = os.path.join(Photospec_dir, f'PhotoSpec_{site_name}.csv')

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


    MCD19_prefill_file = os.path.join(dat_pt_basedir_prefill, f'MCD19_QCfilt_{QC_descr}_{site_name}.nc')
    MCD19_prefill_OutlierFilter_file = os.path.join(dat_pt_basedir_prefill_OutlierFilter, f'MCD19_QCfilt_{QC_descr}_{site_name}.nc')
    # climatology shown in the time-series plots: post-outlier-filtering historical average
    MCD19_histavg_file = os.path.join(dat_pt_basedir_hist_avg_OutlierFilter, f'MCD19_QCfilt_{QC_descr}_hist_avg_OutlierFilter_{site_name}.nc')
    MCD19_lmfill_file = os.path.join(dat_pt_basedir_lmfill, f'MCD19_QCfilt_{QC_descr}_{site_name}.nc')
    MCD19_basicfill_file = os.path.join(dat_pt_basedir_basicfill, f'MCD19_QCfilt_{QC_descr}_{site_name}.nc')



    MCD19_prefill_pt = xr.open_dataset(MCD19_prefill_file)
    MCD19_prefill_OutlierFilter_pt = xr.open_dataset(MCD19_prefill_OutlierFilter_file)
    MCD19_histavg_lookup = xr.open_dataset(MCD19_histavg_file)
    MCD19_lmfill_pt = xr.open_dataset(MCD19_lmfill_file)
    MCD19_basicfill_pt = xr.open_dataset(MCD19_basicfill_file)

    # Use the lm-fill dataset as the template, then populate each time step from
    # the historical-average lookup table using its day-of-year coordinate.
    MCD19_histavg_pt = MCD19_lmfill_pt.copy(deep=True)
    for variable in MCD19_histavg_pt.data_vars:
        MCD19_histavg_pt[variable] = xr.full_like(
            MCD19_histavg_pt[variable], np.nan, dtype=float
        )

    target_doy = xr.DataArray(
        pd.DatetimeIndex(MCD19_histavg_pt.time.values).dayofyear,
        dims='time',
        coords={'time': MCD19_histavg_pt.time},
    )

    for variable in MCD19_histavg_lookup.data_vars:
        if 'doy' in MCD19_histavg_lookup.dims:
            print(f"Populating {variable} from historical average lookup table using day-of-year coordinate.")
            MCD19_histavg_pt[variable] = (
                MCD19_histavg_lookup[variable]
                .sel(doy=target_doy)
                .assign_coords(time=MCD19_histavg_pt.time)
            )


    #%%

    Photospec_df = pd.read_csv(Photospec_file)
    print("Photospec_df columns:", Photospec_df.columns.tolist())

    # Remove any data from Photospec_df where NDVI is below 0.5
    Photospec_df = Photospec_df[Photospec_df['NDVI'] >= 0.5].reset_index(drop=True)
    # Convert Photospec timestamps from "M/D/YY HH:MM" format to datetime64
    Photospec_times = pd.to_datetime(Photospec_df['Time'], format='%m/%d/%y %H:%M')

    # Select MCD19 data within the time range of Photospec data
    MCD19_prefill_pt = MCD19_prefill_pt.sel(time=slice(
        Photospec_times.iloc[0] - pd.Timedelta(days=16),
        Photospec_times.iloc[-1] + pd.Timedelta(days=16),
    ))
    MCD19_prefill_OutlierFilter_pt = MCD19_prefill_OutlierFilter_pt.sel(time=slice(
        Photospec_times.iloc[0] - pd.Timedelta(days=16),
        Photospec_times.iloc[-1] + pd.Timedelta(days=16),
    ))
    MCD19_histavg_pt = MCD19_histavg_pt.sel(time=slice(
        Photospec_times.iloc[0] - pd.Timedelta(days=16),
        Photospec_times.iloc[-1] + pd.Timedelta(days=16),
    ))
    MCD19_lmfill_pt = MCD19_lmfill_pt.sel(time=slice(
        Photospec_times.iloc[0] - pd.Timedelta(days=16),
        Photospec_times.iloc[-1] + pd.Timedelta(days=16),
    ))
    MCD19_basicfill_pt = MCD19_basicfill_pt.sel(time=slice(
        Photospec_times.iloc[0] - pd.Timedelta(days=16),
        Photospec_times.iloc[-1] + pd.Timedelta(days=16),
    ))


    MCD19_prefill_times = np.array([np.datetime64(t) for t in MCD19_prefill_pt['time'].values])
    MCD19_prefill_OutlierFilter_times = np.array([np.datetime64(t) for t in MCD19_prefill_OutlierFilter_pt['time'].values])
    MCD19_histavg_times = np.array([np.datetime64(t) for t in MCD19_histavg_pt['time'].values])
    MCD19_lmfill_times = np.array([np.datetime64(t) for t in MCD19_lmfill_pt['time'].values])
    MCD19_basicfill_times = np.array([np.datetime64(t) for t in MCD19_basicfill_pt['time'].values])


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
       'axes.titlesize': 19,
       'xtick.labelsize': 15,
       'ytick.labelsize': 15,
       'legend.fontsize': 13
    })

    MCD19_prefill_color = '#6EF0FF'
    MCD19_prefill_OutlierFilter_color = '#5500FF' 
    MCD19_histavg_color = '#FFC763'   
    MCD19_basicfill_color = '#FF8C00' 
    MCD19_lmfill_color = '#BF2665'

    # Plot shaded regions for winter periods
    winter_periods = [
        # Ground snow cover periods
    ]




    ############################################
    ### QC ANALYSIS DELIVERABLE 1 - NDVI time series v2 (two axes)
    ############################################


    # Calculate day of year for MCD19_CloudFree_times and Photospec_times_filtered

    ndvi_arrays = [
        MCD19_prefill_pt['NDVI'].values,
        MCD19_prefill_OutlierFilter_pt['NDVI'].values,
        MCD19_histavg_pt['NDVI'].values,
        MCD19_lmfill_pt['NDVI'].values,
        MCD19_basicfill_pt['NDVI'].values,
    ]
    mcd19_ndvi_min = np.nanmin(np.concatenate(ndvi_arrays))
    mcd19_ndvi_max = np.nanmax(np.concatenate(ndvi_arrays))
    mcd19_ndvi_range = mcd19_ndvi_max - mcd19_ndvi_min


    fig, ax = plt.subplots(figsize=(7, 3))

    for start, end in winter_periods:
        ax.axvspan(start, end, color='lightblue', alpha=0.5)
    ax.scatter(Photospec_times_filtered, Photospec_df_filtered['NDVI'], color='black', s=10, alpha=0.7)
    ax2 = ax.twinx()


    ax2.plot(MCD19_histavg_times, MCD19_histavg_pt['NDVI'].values, color=MCD19_histavg_color, marker = '^', markersize=5, linewidth=1.5, alpha=1, markeredgecolor='none')
    ax2.fill_between(MCD19_histavg_times,
                    MCD19_histavg_pt['NDVI'].values - MCD19_histavg_pt['NDVI_std'].values,
                    MCD19_histavg_pt['NDVI'].values + MCD19_histavg_pt['NDVI_std'].values,
                    color=MCD19_histavg_color, alpha=0.5, zorder = 2)


    ax2.plot(MCD19_prefill_times, MCD19_prefill_pt['NDVI'].values, color=MCD19_prefill_color, marker = 'o', markersize=6, linewidth=2, alpha=1, markeredgecolor='black', markeredgewidth=0.5)
    ax2.plot(MCD19_prefill_OutlierFilter_times, MCD19_prefill_OutlierFilter_pt['NDVI'].values, color=MCD19_prefill_OutlierFilter_color, marker = 'o', markersize=6, linewidth=2, alpha=1, markeredgecolor='black', markeredgewidth=0.5)
    # ax2.scatter(MCD19_histavg_times, MCD19_histavg_pt['NDVI'].values, color=MCD19_histavg_color, marker = '^', s=80, alpha=0.7, edgecolor='black', linewidth=0.5)

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
    ndvi_panel_file = os.path.join(panel_dir, f'{site_name}_NDVI_OutlierFilter_panel.png')
    plt.savefig(ndvi_panel_file, dpi=300, bbox_inches='tight')
    


    #%%


    ############################################
    ### QC ANALYSIS DELIVERABLE 1 - CCI time series v2 (two axes)
    ############################################


    # Calculate day of year for MCD19_CloudFree_times and Photospec_times_filtered

    CCI_arrays = [
        MCD19_prefill_pt['CCI'].values,
        MCD19_prefill_OutlierFilter_pt['CCI'].values,
        MCD19_histavg_pt['CCI'].values,
        MCD19_lmfill_pt['CCI'].values,
        MCD19_basicfill_pt['CCI'].values,
    ]
    mcd19_CCI_min = np.nanmin(np.concatenate(CCI_arrays))
    mcd19_CCI_max = np.nanmax(np.concatenate(CCI_arrays))
    mcd19_CCI_range = mcd19_CCI_max - mcd19_CCI_min


    fig, ax = plt.subplots(figsize=(7, 3))

    for start, end in winter_periods:
        ax.axvspan(start, end, color='lightblue', alpha=0.5)
    ax.scatter(Photospec_times_filtered, Photospec_df_filtered['CCI'], color='black', s=10, alpha=0.7)
    ax2 = ax.twinx()




    ax2.plot(MCD19_histavg_times, MCD19_histavg_pt['CCI'].values, color=MCD19_histavg_color, marker = '^', markersize=5, linewidth=1.5, alpha=1, markeredgecolor='none')
    ax2.fill_between(MCD19_histavg_times,
                    MCD19_histavg_pt['CCI'].values - MCD19_histavg_pt['CCI_std'].values,
                    MCD19_histavg_pt['CCI'].values + MCD19_histavg_pt['CCI_std'].values,
                    color=MCD19_histavg_color, alpha=0.5, zorder = 2)


    ax2.plot(MCD19_prefill_times, MCD19_prefill_pt['CCI'].values, color=MCD19_prefill_color, marker = 'o', markersize=6, linewidth=2, alpha=1, markeredgecolor='black', markeredgewidth=0.5)
    ax2.plot(MCD19_prefill_OutlierFilter_times, MCD19_prefill_OutlierFilter_pt['CCI'].values, color=MCD19_prefill_OutlierFilter_color, marker = 'o', markersize=6, linewidth=2, alpha=1, markeredgecolor='black', markeredgewidth=0.5)
    # ax2.scatter(MCD19_histavg_times, MCD19_histavg_pt['CCI'].values, color=MCD19_histavg_color, marker = '^', s=80, alpha=0.7, edgecolor='black', linewidth=0.5)

    # ax2.plot(MCD19_lmfill_times, MCD19_lmfill_pt['CCI'].values, color=MCD19_lmfill_color, marker = 'o', markersize=6, linewidth=2, alpha=1, markeredgecolor='black', markeredgewidth=0.5)
    # ax2.plot(MCD19_basicfill_times, MCD19_basicfill_pt['CCI'].values, color=MCD19_basicfill_color, marker = 'o', markersize=6, linewidth=2, alpha=1, markeredgecolor='black', markeredgewidth=0.5)

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
    cci_panel_file = os.path.join(panel_dir, f'{site_name}_CCI_OutlierFilter_panel.png')
    plt.savefig(cci_panel_file, dpi=300, bbox_inches='tight')
    

    ############################################
    ### QC ANALYSIS DELIVERABLE 1 - NDVI time series v2 (two axes)
    ### Gap-filling comparison
    ############################################


    # Calculate day of year for MCD19_CloudFree_times and Photospec_times_filtered

    ndvi_arrays = [
        MCD19_prefill_pt['NDVI'].values,
        MCD19_prefill_OutlierFilter_pt['NDVI'].values,
        MCD19_histavg_pt['NDVI'].values,
        MCD19_lmfill_pt['NDVI'].values,
        MCD19_basicfill_pt['NDVI'].values,
    ]
    mcd19_ndvi_min = np.nanmin(np.concatenate(ndvi_arrays))
    mcd19_ndvi_max = np.nanmax(np.concatenate(ndvi_arrays))
    mcd19_ndvi_range = mcd19_ndvi_max - mcd19_ndvi_min


    fig, ax = plt.subplots(figsize=(7, 3))

    for start, end in winter_periods:
        ax.axvspan(start, end, color='lightblue', alpha=0.5)
    ax.scatter(Photospec_times_filtered, Photospec_df_filtered['NDVI'], color='black', s=10, alpha=0.7)
    ax2 = ax.twinx()


    ax2.plot(MCD19_histavg_times, MCD19_histavg_pt['NDVI'].values, color=MCD19_histavg_color, marker = '^', markersize=5, linewidth=1.5, alpha=1, markeredgecolor='none')
    ax2.fill_between(MCD19_histavg_times,
                    MCD19_histavg_pt['NDVI'].values - MCD19_histavg_pt['NDVI_std'].values,
                    MCD19_histavg_pt['NDVI'].values + MCD19_histavg_pt['NDVI_std'].values,
                    color=MCD19_histavg_color, alpha=0.5, zorder = 2)


    # ax2.plot(MCD19_prefill_times, MCD19_prefill_pt['NDVI'].values, color=MCD19_prefill_color, marker = 'o', markersize=6, linewidth=2, alpha=1, markeredgecolor='black', markeredgewidth=0.5)

    ax2.plot(MCD19_lmfill_times, MCD19_lmfill_pt['NDVI'].values, color=MCD19_lmfill_color, marker = 'o', markersize=6, linewidth=2, alpha=1, markeredgecolor='black', markeredgewidth=0.5)
    ax2.plot(MCD19_basicfill_times, MCD19_basicfill_pt['NDVI'].values, color=MCD19_basicfill_color, marker = 'o', markersize=6, linewidth=2, alpha=1, markeredgecolor='black', markeredgewidth=0.5)
    ax2.plot(MCD19_prefill_OutlierFilter_times, MCD19_prefill_OutlierFilter_pt['NDVI'].values, color=MCD19_prefill_OutlierFilter_color, marker = 'o', markersize=6, linewidth=2, alpha=1, markeredgecolor='black', markeredgewidth=0.5)

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
    ndvi_panel_file = os.path.join(panel_dir, f'{site_name}_NDVI_GFcompare_panel.png')
    plt.savefig(ndvi_panel_file, dpi=300, bbox_inches='tight')
    


    #%%


    ############################################
    ### QC ANALYSIS DELIVERABLE 1 - CCI time series v2 (two axes)
    ### Gap-filling comparison
    ############################################


    # Calculate day of year for MCD19_CloudFree_times and Photospec_times_filtered

    CCI_arrays = [
        MCD19_prefill_pt['CCI'].values,
        MCD19_prefill_OutlierFilter_pt['CCI'].values,
        MCD19_histavg_pt['CCI'].values,
        MCD19_lmfill_pt['CCI'].values,
        MCD19_basicfill_pt['CCI'].values,
    ]
    mcd19_CCI_min = np.nanmin(np.concatenate(CCI_arrays))
    mcd19_CCI_max = np.nanmax(np.concatenate(CCI_arrays))
    mcd19_CCI_range = mcd19_CCI_max - mcd19_CCI_min


    fig, ax = plt.subplots(figsize=(7, 3))

    for start, end in winter_periods:
        ax.axvspan(start, end, color='lightblue', alpha=0.5)
    ax.scatter(Photospec_times_filtered, Photospec_df_filtered['CCI'], color='black', s=10, alpha=0.7)
    ax2 = ax.twinx()




    ax2.plot(MCD19_histavg_times, MCD19_histavg_pt['CCI'].values, color=MCD19_histavg_color, marker = '^', markersize=5, linewidth=1.5, alpha=1, markeredgecolor='none')
    ax2.fill_between(MCD19_histavg_times,
                    MCD19_histavg_pt['CCI'].values - MCD19_histavg_pt['CCI_std'].values,
                    MCD19_histavg_pt['CCI'].values + MCD19_histavg_pt['CCI_std'].values,
                    color=MCD19_histavg_color, alpha=0.5, zorder = 2)



    ax2.plot(MCD19_lmfill_times, MCD19_lmfill_pt['CCI'].values, color=MCD19_lmfill_color, marker = 'o', markersize=6, linewidth=2, alpha=1, markeredgecolor='black', markeredgewidth=0.5)
    ax2.plot(MCD19_basicfill_times, MCD19_basicfill_pt['CCI'].values, color=MCD19_basicfill_color, marker = 'o', markersize=6, linewidth=2, alpha=1, markeredgecolor='black', markeredgewidth=0.5)
    ax2.plot(MCD19_prefill_OutlierFilter_times, MCD19_prefill_OutlierFilter_pt['CCI'].values, color=MCD19_prefill_OutlierFilter_color, marker = 'o', markersize=6, linewidth=2, alpha=1, markeredgecolor='black', markeredgewidth=0.5)


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
    cci_panel_file = os.path.join(panel_dir, f'{site_name}_CCI_GFcompare_panel.png')
    plt.savefig(cci_panel_file, dpi=300, bbox_inches='tight')


    ############################################
    ### Historical average vs. outlier-filtered historical average
    ### (NDVI and CCI climatology by day of year, +/- 1 std)
    ############################################

    MCD19_histavg_OutlierFilter_file = os.path.join(dat_pt_basedir_hist_avg_OutlierFilter, f'MCD19_QCfilt_{QC_descr}_hist_avg_OutlierFilter_{site_name}.nc')
    MCD19_histavg_clim = xr.open_dataset(os.path.join(dat_pt_basedir_hist_avg, f'MCD19_QCfilt_{QC_descr}_hist_avg_{site_name}.nc'))
    MCD19_histavg_OutlierFilter_clim = xr.open_dataset(MCD19_histavg_OutlierFilter_file)

    # (dataset, color, linewidth, markersize) - the unfiltered series is drawn wider
    # underneath so it stays visible where the two climatologies overlap
    histavg_series = [
        (MCD19_histavg_clim, MCD19_histavg_color, 3.5, 8),
        (MCD19_histavg_OutlierFilter_clim, MCD19_prefill_OutlierFilter_color, 1.5, 5),
    ]

    for var_name, panel_letter in [('NDVI', 'a'), ('CCI', 'b')]:

        fig, ax = plt.subplots(figsize=(7, 3))

        for clim_ds, color, lw, ms in histavg_series:
            ax.plot(clim_ds['doy'].values, clim_ds[var_name].values, color=color, marker='^', markersize=ms, linewidth=lw, markeredgecolor='none', zorder=3)
            ax.fill_between(clim_ds['doy'].values,
                            clim_ds[var_name].values - clim_ds[f'{var_name}_std'].values,
                            clim_ds[var_name].values + clim_ds[f'{var_name}_std'].values,
                            color=color, alpha=0.35, linewidth=0, zorder=2)
            # dashed lines along the upper/lower std bounds to make the band edges visible
            for sign in (1, -1):
                ax.plot(clim_ds['doy'].values,
                        clim_ds[var_name].values + sign * clim_ds[f'{var_name}_std'].values,
                        color=color, linestyle='--', linewidth=0.6 * lw, zorder=2.5)

            # fill_between draws nothing for isolated points (both neighbors NaN),
            # so show their std as error bars instead
            vals = clim_ds[var_name].values
            valid = np.isfinite(vals)
            prev_valid = np.concatenate([[False], valid[:-1]])
            next_valid = np.concatenate([valid[1:], [False]])
            lone = valid & ~prev_valid & ~next_valid
            if lone.any():
                ax.errorbar(clim_ds['doy'].values[lone], vals[lone],
                            yerr=clim_ds[f'{var_name}_std'].values[lone],
                            fmt='none', ecolor=color, alpha=0.5, elinewidth=lw + 1, capsize=4, capthick=lw, zorder=2)

        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

        ax.set_xlim(1, 366)
        ax.set_xticks(month_starts)
        ax.set_xticklabels(month_labels)
        ax.set_title(f'{panel_letter})  {plot_title}')
        ax.set_ylabel(f'{var_name} (MCD19)')

        plt.tight_layout()
        histavg_panel_file = os.path.join(panel_dir, f'{site_name}_{var_name}_HistAvgCompare_panel.png')
        plt.savefig(histavg_panel_file, dpi=300, bbox_inches='tight')
        # plt.show()


if __name__ == "__main__":
    main()
