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
#%%
# use Google Satellite imagery as basemap
tiler = img_tiles.GoogleTiles(style='satellite')
crs = tiler.crs # set crs of map tiler
alpha = 0.8  # transparency 0-1
transform = ccrs.PlateCarree()  # transform specifies the crs that the data is in

### Load state boundaries
fn = shpreader.natural_earth(
   resolution='10m', category='cultural', 
   name='admin_1_states_provinces',
)

reader = shpreader.Reader(fn)
states = [x for x in reader.records() if x.attributes["admin"] == "United States of America"] # get all states in US
states_geom = cfeature.ShapelyFeature([x.geometry for x in states], ccrs.PlateCarree())

# ### Create a custom colormap
# plot_colors = ["#ffffb2", "#fd8d3c", "#f03b20", "#810f7c", "#21013B"]
# cmap = mcolors.ListedColormap(plot_colors)
# cmaplist = [cmap(i) for i in range(cmap.N)] # extract all colors from the cmap

# # create the new map
# cmap = mcolors.LinearSegmentedColormap.from_list(
#     'Custom cmap', cmaplist, cmap.N)

tiler_zoom = 5  # define a zoom level of detail

def get_tile_plot_extent(dat_xr_curv):
    lat_min = float(dat_xr_curv['lat'].min())
    lat_max = float(dat_xr_curv['lat'].max())
    lon_min = float(dat_xr_curv['lon'].min())
    lon_max = float(dat_xr_curv['lon'].max())
    plot_extent =[lon_min, lon_max, lat_min, lat_max]

    # special case if some lon values cross the 180 meridian, then lon_max will be 
    # large because it extends into Russia (lon ~ +100 to +180)
    if ((lon_min < -175) & (lon_max > 100)):
        lon_neg = dat_xr_curv.lon.values.copy()
        lon_neg[lon_neg > 0] = np.nan
        plot_extent = [-179.99, np.nanmax(lon_neg), lat_min, lat_max]

    return plot_extent


def test_plot(dat_da, plot_title, zlab, plot_extent = None, plot_filename = None, plot_min = None, plot_max = None):
    if plot_extent is None:
        plot_extent = get_tile_plot_extent(dat_da)
    if plot_min is None:
        plot_min = float(dat_da.nanmin())
    if plot_max is None:
        plot_max = float(dat_da.nanmax())
    fig, ax = plt.subplots(figsize=(15, 15), subplot_kw={'projection': crs}) # establish the figure, axes
    ax.set_extent(plot_extent, crs=ccrs.PlateCarree())  # PlateCarree is default lat/lon    
    ax.add_image(tiler, tiler_zoom, interpolation='none') # Add base image
    ax.add_feature(states_geom, facecolor="none", edgecolor="black", linewidth=1) # add state boundaries to the map
    dat_da.plot(ax=ax, x='lon', y='lat', transform=transform, 
                    alpha=alpha, cmap='inferno_r',
                    vmin=plot_min, vmax=plot_max,
                    cbar_kwargs={'orientation': 'vertical',
                                            'pad': 0.05,
                                            'label': zlab,
                                            'shrink': 0.75})

    plt.title(plot_title, fontsize=18)
    plt.xlabel('lon', fontsize=15)
    plt.ylabel('lat', fontsize=14)

    if plot_filename is not None:
        if not os.path.exists(os.path.dirname(plot_filename)):
            Path(os.path.dirname(plot_filename)).mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_filename, format = "png", bbox_inches = "tight")
        plt.close('all')
    else:
        plt.show()

def clip_2dcurv_ds_to_point(ds, lon_pt, lat_pt):
    # Find the x, y indices where lon_pt and lat_pt are found in ds
    lon_2d = ds.lon.values
    lat_2d = ds.lat.values

    # Compute the absolute difference and find the indices of the minimum
    dist = np.abs(lon_2d - lon_pt) + np.abs(lat_2d - lat_pt)
    y_idx, x_idx = np.unravel_index(np.argmin(dist), lon_2d.shape)
    lon_selected = ds.lon.isel(x=x_idx, y=y_idx).values
    lat_selected = ds.lat.isel(x=x_idx, y=y_idx).values
    print(f"Nearest point in dataset to ({lon_pt}, {lat_pt}) is at ({lon_selected}, {lat_selected}) with distance {dist[y_idx, x_idx]:.6f}")
    ds_point = ds.isel(x=x_idx, y=y_idx)
    return ds_point


def detect_outliers_mad(ndvi_arr, cci_arr, k=2.5, min_valid_count=3):
    """
    Detect outliers in NDVI and CCI using Median Absolute Deviation.
    
    Also flags entire pixels as outliers if they have fewer than min_valid_count
    observations, ensuring the threshold is enforced via the mask.
    
    Parameters:
    -----------
    ndvi_arr, cci_arr : array-like
        Arrays of any shape with time as first dimension.
    k : float, default 2.5
        Scaling factor for MAD threshold
    min_valid_count : int, default 3
        Minimum valid points required per pixel/location
    
    Returns:
    --------
    outlier_mask : ndarray of bool
        Same shape as input. True where point should be excluded:
        - MAD-detected outliers, OR
        - ALL observations at pixels with < min_valid_count valid data
    stats : dict
        Summary statistics
    """
    
    ndvi_arr = np.asarray(ndvi_arr)
    cci_arr = np.asarray(cci_arr)
    
    original_shape = ndvi_arr.shape
    
    # Reshape to (time, n_pixels)
    ndvi_reshaped = ndvi_arr.reshape(ndvi_arr.shape[0], -1)
    cci_reshaped = cci_arr.reshape(cci_arr.shape[0], -1)
    
    n_time, n_pixels = ndvi_reshaped.shape
    
    # Count valid points per pixel
    valid_per_pixel = np.sum(~(np.isnan(ndvi_reshaped) | np.isnan(cci_reshaped)), axis=0)
    is_valid_per_pixel = valid_per_pixel >= min_valid_count
    
    # Initialize outlier mask
    outlier_mask = np.zeros_like(ndvi_reshaped, dtype=bool)
    
    # Apply MAD per pixel
    for pix in range(n_pixels):
        # If insufficient data, flag ALL points at this pixel as outliers
        if not is_valid_per_pixel[pix]:
            outlier_mask[:, pix] = True
            continue
        
        ndvi_col = ndvi_reshaped[:, pix]
        cci_col = cci_reshaped[:, pix]
        
        # Get valid values for this pixel
        valid_mask = ~(np.isnan(ndvi_col) | np.isnan(cci_col))
        ndvi_valid = ndvi_col[valid_mask]
        cci_valid = cci_col[valid_mask]
        
        # NDVI MAD
        ndvi_median = np.median(ndvi_valid)
        ndvi_mad = np.median(np.abs(ndvi_valid - ndvi_median))
        if ndvi_mad > 0:
            ndvi_outliers = np.abs(ndvi_col - ndvi_median) > (k * ndvi_mad)
        else:
            ndvi_outliers = np.zeros_like(ndvi_col, dtype=bool)
        
        # CCI MAD
        cci_median = np.median(cci_valid)
        cci_mad = np.median(np.abs(cci_valid - cci_median))
        if cci_mad > 0:
            cci_outliers = np.abs(cci_col - cci_median) > (k * cci_mad)
        else:
            cci_outliers = np.zeros_like(cci_col, dtype=bool)
        
        # Flag if outlier in either index
        pixel_outliers = ndvi_outliers | cci_outliers
        outlier_mask[:, pix] = pixel_outliers
    
    # Reshape back to original shape
    outlier_mask = outlier_mask.reshape(original_shape)
    
    # Summary stats
    n_valid_total = np.sum(~(np.isnan(ndvi_arr) | np.isnan(cci_arr)))
    n_outliers_total = np.sum(outlier_mask)
    n_pixels_insufficient = np.sum(~is_valid_per_pixel)
    
    stats = {
        'n_valid_total': n_valid_total,
        'n_outliers_total': n_outliers_total,
        'n_pixels_valid': np.sum(is_valid_per_pixel),
        'n_pixels_insufficient': n_pixels_insufficient,
        'min_valid_count': min_valid_count,
    }
    
    return outlier_mask, stats

#########################################
# Define Global Variables and constants
#########################################

hist_years = np.arange(2000, 2025)
DOY_composite_starts = np.arange(1, 366, 16)
hist_composites = range(len(DOY_composite_starts))  # 0-22


# Site coordinates, date ranges, and save paths
site_info = {
    'US-NR1': {
        'coords': (-105.5464, 40.0329),
        'start': '2017-01-01',
        'end': '2018-12-31',
        'tile': 'h09v04'
    },
    'OSBS': {
        'coords': (-81.993431, 29.689282),
        'start': '2021-01-01',
        'end': '2022-12-31',
        'tile': 'h10v06'
    },
    'Ca-Obs': {
        'coords': (-105.1178, 53.9872),
        'start': '2018-01-01',
        'end': '2022-12-31',
        'tile': 'h11v03'
    },
    'DEJU': {
        'coords': (-145.75136, 63.88112),
        'start': '2019-01-01',
        'end': '2021-12-31',
        'tile': 'h11v02'
    },
    'US-UMB': {
        'coords': (-84.7138, 45.5598),
        'start': '2018-01-01',
        'end': '2019-12-31',
        'tile': 'h12v04'
    },
    'US-Ne3': {
        'coords': (-96.4397, 41.1797),
        'start': '2018-01-01',
        'end': '2019-12-31',
        'tile': 'h10v04'
    }
}
#########################################
# Begin main
#########################################
#%%

# mark start time to keep track of elapsed
start_total = time.time()

site = 'US-NR1'
MODIS_tile = 'h09v04'
QC_option = 'CloudFree_LowAOD_ClearAdj' #

dat_dir = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/pre-fill/QCfilt_{QC_option}/{MODIS_tile}/'
climatology_dir = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/pre-fill/QCfilt_{QC_option}/{MODIS_tile}/hist_avg/'
plot_dir = f'/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/plots/maps/MAIAC_CV-MVC/1km_curv/hist_avg/QCfilt_{QC_option}/{MODIS_tile}/'
filtered_out_dir = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/pre-fill_OutlierFilter/QCfilt_{QC_option}/{MODIS_tile}/'
out_dir = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/pre-fill/QCfilt_{QC_option}/{MODIS_tile}/hist_avg/'
Path(plot_dir).mkdir(parents=True, exist_ok=True)
Path(out_dir).mkdir(parents=True, exist_ok=True)
Path(filtered_out_dir).mkdir(parents=True, exist_ok=True)

min_valid_count = 3  # minimum number of valid points required to compute mean for a composite period
composite_periods = np.arange(0, 23)  # 0-22
NDVI_arr_dict_all = {cp: {} for cp in composite_periods}
NDVI_arr_dict_filter = {cp: {} for cp in composite_periods}
CCI_arr_dict_all = {cp: {} for cp in composite_periods}
CCI_arr_dict_filter = {cp: {} for cp in composite_periods}

dat_xr_arr = []
climatology_xr_arr = []

composite_year = 2018

# load last two composite periods of the previous year to ensure continuity for outlier detection
if composite_year > 2000:
    for composite_period in [21, 22]:

        # e.g. MCD19A1.h12v04.061_1km_16day_CV-MVC_QCfilt_CloudFree_LowAOD_ClearAdj_2025-19.nc
        dat_file = os.path.join(dat_dir, f'MCD19A1.{MODIS_tile}.061_1km_16day_CV-MVC_QCfilt_{QC_option}_{composite_year-1}-{str(composite_period).zfill(2)}.nc')
        climatology_file = os.path.join(climatology_dir, f'MCD19A1.{MODIS_tile}.061_1km_CV-MVC_QCfilt_{QC_option}_hist_median_{composite_period:02d}.nc')

        # Skip processing if 'v01' is in MODIS_tile and composite_period is in [0, 1, 2, 3, 19, 20, 21, 22]
        if 'v01' in MODIS_tile and composite_period in [0, 1, 2, 3, 19, 20, 21, 22]:
            print(f"Skipping composite_period {composite_period} for MODIS_tile {MODIS_tile} (high latitude winter, no data).")
            # continue

        # Skip processing if 'v02' is in MODIS_tile and composite_period is in [0, 1, 20, 21, 22]
        if 'v02' in MODIS_tile and composite_period in [0, 1, 20, 21, 22]:
            print(f"Skipping composite_period {composite_period} for MODIS_tile {MODIS_tile} (high latitude winter, no data).")
            # continue

        print(f'Loading file: {dat_file}')
        dat_ds = xr.open_dataset(dat_file)
        climatology_ds = xr.open_dataset(climatology_file).expand_dims(doy = [DOY_composite_starts[composite_period]])
        if 'NDVI_rank' in dat_ds:
            dat_ds = dat_ds.drop_vars(['NDVI_rank'])
        if 'VZA_rank' in dat_ds:
            dat_ds = dat_ds.drop_vars(['VZA_rank'])


        dat_xr_pt = clip_2dcurv_ds_to_point(dat_ds, site_info[site]['coords'][0], site_info[site]['coords'][1])
        climatology_xr_pt = clip_2dcurv_ds_to_point(climatology_ds, site_info[site]['coords'][0], site_info[site]['coords'][1])

        climatology_xr_pt.assign_coords(time=dat_xr_pt.time.values)  # Ensure time coordinates match for concatenation

        dat_xr_arr.append(dat_xr_pt)
        climatology_xr_arr.append(climatology_xr_pt)


for composite_period in composite_periods:

    # e.g. MCD19A1.h12v04.061_1km_16day_CV-MVC_QCfilt_CloudFree_LowAOD_ClearAdj_2025-19.nc
    dat_file = os.path.join(dat_dir, f'MCD19A1.{MODIS_tile}.061_1km_16day_CV-MVC_QCfilt_{QC_option}_{composite_year}-{str(composite_period).zfill(2)}.nc')
    climatology_file = os.path.join(climatology_dir, f'MCD19A1.{MODIS_tile}.061_1km_CV-MVC_QCfilt_{QC_option}_hist_median_{composite_period:02d}.nc')

    # Skip processing if 'v01' is in MODIS_tile and composite_period is in [0, 1, 2, 3, 19, 20, 21, 22]
    if 'v01' in MODIS_tile and composite_period in [0, 1, 2, 3, 19, 20, 21, 22]:
        print(f"Skipping composite_period {composite_period} for MODIS_tile {MODIS_tile} (high latitude winter, no data).")
        # continue

    # Skip processing if 'v02' is in MODIS_tile and composite_period is in [0, 1, 20, 21, 22]
    if 'v02' in MODIS_tile and composite_period in [0, 1, 20, 21, 22]:
        print(f"Skipping composite_period {composite_period} for MODIS_tile {MODIS_tile} (high latitude winter, no data).")
        # continue

    print(f'Loading file: {dat_file}')
    dat_ds = xr.open_dataset(dat_file)
    climatology_ds = xr.open_dataset(climatology_file).expand_dims(doy = [DOY_composite_starts[composite_period]])
    if 'NDVI_rank' in dat_ds:
        dat_ds = dat_ds.drop_vars(['NDVI_rank'])
    if 'VZA_rank' in dat_ds:
        dat_ds = dat_ds.drop_vars(['VZA_rank'])

    dat_xr_pt = clip_2dcurv_ds_to_point(dat_ds, site_info[site]['coords'][0], site_info[site]['coords'][1])
    climatology_xr_pt = clip_2dcurv_ds_to_point(climatology_ds, site_info[site]['coords'][0], site_info[site]['coords'][1])

    climatology_xr_pt.assign_coords(time=dat_xr_pt.time.values)  # Ensure time coordinates match for concatenation

    dat_xr_arr.append(dat_xr_pt)
    climatology_xr_arr.append(climatology_xr_pt)


# load first two composite periods of the next year to ensure continuity for outlier detection
if composite_year < 2025:
    for composite_period in [0, 1]:

        # e.g. MCD19A1.h12v04.061_1km_16day_CV-MVC_QCfilt_CloudFree_LowAOD_ClearAdj_2025-19.nc
        dat_file = os.path.join(dat_dir, f'MCD19A1.{MODIS_tile}.061_1km_16day_CV-MVC_QCfilt_{QC_option}_{composite_year+1}-{str(composite_period).zfill(2)}.nc')
        climatology_file = os.path.join(climatology_dir, f'MCD19A1.{MODIS_tile}.061_1km_CV-MVC_QCfilt_{QC_option}_hist_median_{composite_period:02d}.nc')

        # Skip processing if 'v01' is in MODIS_tile and composite_period is in [0, 1, 2, 3, 19, 20, 21, 22]
        if 'v01' in MODIS_tile and composite_period in [0, 1, 2, 3, 19, 20, 21, 22]:
            print(f"Skipping composite_period {composite_period} for MODIS_tile {MODIS_tile} (high latitude winter, no data).")
            # continue

        # Skip processing if 'v02' is in MODIS_tile and composite_period is in [0, 1, 20, 21, 22]
        if 'v02' in MODIS_tile and composite_period in [0, 1, 20, 21, 22]:
            print(f"Skipping composite_period {composite_period} for MODIS_tile {MODIS_tile} (high latitude winter, no data).")
            # continue

        print(f'Loading file: {dat_file}')
        dat_ds = xr.open_dataset(dat_file)
        climatology_ds = xr.open_dataset(climatology_file).expand_dims(doy = [DOY_composite_starts[composite_period]])
        if 'NDVI_rank' in dat_ds:
            dat_ds = dat_ds.drop_vars(['NDVI_rank'])
        if 'VZA_rank' in dat_ds:
            dat_ds = dat_ds.drop_vars(['VZA_rank'])


        dat_xr_pt = clip_2dcurv_ds_to_point(dat_ds, site_info[site]['coords'][0], site_info[site]['coords'][1])
        climatology_xr_pt = clip_2dcurv_ds_to_point(climatology_ds, site_info[site]['coords'][0], site_info[site]['coords'][1])

        climatology_xr_pt.assign_coords(time=dat_xr_pt.time.values)  # Ensure time coordinates match for concatenation
        dat_xr_arr.append(dat_xr_pt)
        climatology_xr_arr.append(climatology_xr_pt)



dat_xr_pt = xr.concat(dat_xr_arr, dim='time')
climatology_xr_pt = xr.concat(climatology_xr_arr, dim='time')








# dat_xr_pt = clip_2dcurv_ds_to_point(dat_xr, site_info[site]['coords'][0], site_info[site]['coords'][1])
# climatology_xr_pt = clip_2dcurv_ds_to_point(climatology_xr, site_info[site]['coords'][0], site_info[site]['coords'][1])


# Use the lm-fill dataset as the template, then populate each time step from
# the historical-average lookup table using its day-of-year coordinate.
MCD19_histavg_pt = dat_xr_pt.copy(deep=True)
for variable in climatology_xr_pt.data_vars:
    MCD19_histavg_pt[variable] = xr.full_like(
        MCD19_histavg_pt['NDVI'], np.nan, dtype=float
    )

target_doy = xr.DataArray(
    pd.DatetimeIndex(MCD19_histavg_pt.time.values).dayofyear,
    dims='time',
    coords={'time': MCD19_histavg_pt.time},
)

for variable in climatology_xr_pt.data_vars:
    if 'doy' in climatology_xr_pt.dims:
        print(f"Populating {variable} from historical average lookup table using day-of-year coordinate.")
        MCD19_histavg_pt[variable] = (
            climatology_xr_pt[variable]
            .sel(doy=target_doy)
            .assign_coords(time=MCD19_histavg_pt.time)
        )



fig, ax = plt.subplots(figsize=(7, 3.8))

ax.plot(MCD19_histavg_pt['time'].values, MCD19_histavg_pt['CCI'].values, color='gray', marker = '^', markersize=5, linewidth=1.5, alpha=1, markeredgecolor='none')
ax.fill_between(MCD19_histavg_pt['time'].values,
                MCD19_histavg_pt['CCI'].values - 3*(MCD19_histavg_pt['CCI_MAD'].values),
                MCD19_histavg_pt['CCI'].values + 3*(MCD19_histavg_pt['CCI_MAD'].values),
                color='gray', alpha=0.3, zorder = 2)

ax.fill_between(MCD19_histavg_pt['time'].values,
                MCD19_histavg_pt['CCI'].values - MCD19_histavg_pt['CCI_MAD'].values,
                MCD19_histavg_pt['CCI'].values + MCD19_histavg_pt['CCI_MAD'].values,
                color='gray', alpha=0.5, zorder = 2)

ax.plot(dat_xr_pt['time'].values, dat_xr_pt['CCI'].values, color='red', marker = 'o', markersize=6, linewidth=2, alpha=1, markeredgecolor='black', markeredgewidth=0.5)

# ax.set_ylim(mcd19_cci_min - (mcd19_cci_range*0.05), mcd19_cci_max + (mcd19_cci_range*0.05))
# ax2.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1])
ax.yaxis.set_major_formatter(FFmt(lambda y, _: f'{y:.1f}'))
# ax.yaxis.set_major_locator(MultipleLocator(0.2))

ax.xaxis.set_major_locator(MonthLocator(bymonth=[1, 7]))
ax.xaxis.set_major_formatter(DateFormatter('%b\n%Y'))
# ax.set_title(f'a)  {plot_title}')
ax.set_ylabel('CCI (MCD19)')

plt.tight_layout()
# plt.savefig(ndvi_panel_file, dpi=300, bbox_inches='tight')
plt.show()



#%%

#########################################
# Step 1: check the concatenated objects
#########################################
print(dat_xr_pt.sizes)
print(climatology_xr_pt.sizes)
print(list(climatology_xr_pt.data_vars))

# year and composite period of each timestep, same order as the loading loops above
step_years = np.array(([composite_year - 1] * 2 if composite_year > 2000 else [])
                      + [composite_year] * len(composite_periods)
                      + ([composite_year + 1] * 2 if composite_year < 2025 else []))
step_cps = np.array(([21, 22] if composite_year > 2000 else [])
                    + list(composite_periods)
                    + ([0, 1] if composite_year < 2025 else []))
assert len(step_cps) == dat_xr_pt.sizes['time'] == climatology_xr_pt.sizes['time']
is_target = step_years == composite_year



#%%
#########################################
# Step 2: NDVI and CCI for each timestep
#########################################
# dat_xr_pt['NDVI'] = (dat_xr_pt['Sur_refl2'] - dat_xr_pt['Sur_refl1']) / (dat_xr_pt['Sur_refl2'] + dat_xr_pt['Sur_refl1'])
# dat_xr_pt['CCI'] = (dat_xr_pt['Sur_refl11'] - dat_xr_pt['Sur_refl1']) / (dat_xr_pt['Sur_refl11'] + dat_xr_pt['Sur_refl1'])

#%%
#########################################
# Step 3: median/MAD matched to each timestep
#########################################
# (median, MAD) variable names in the hist_median files, replace with your actual names
CLIM_VARS = {'NDVI': ('NDVI_median', 'NDVI_MAD'),
                'CCI':  ('CCI_median',  'CCI_MAD')}

def clim_at_steps(var):
    da = climatology_xr_pt[var]
    if 'doy' in da.dims:  # outer-joined doy: pick the doy belonging to each timestep
        return np.array([da.isel(time=i).sel(doy=DOY_composite_starts[cp]).item()
                         for i, cp in enumerate(step_cps)])
    return da.values





#%%
#########################################
# Step 4: flag exceedances of median +/- 3*MAD, with direction
#########################################
K = 3.0
MAD_SCALE = 1.0   # 1.0 = raw 3*MAD as specified; 1.4826 would make this ~3 sigma
MAD_FLOOR = 0.0   # > 0 keeps MAD = 0 pixels from flagging every deviation

for idx, (med_var, mad_var) in CLIM_VARS.items():
    x = dat_xr_pt[idx].values
    med = clim_at_steps(med_var)
    mad = clim_at_steps(mad_var)
    thr = K * np.maximum(MAD_SCALE * mad, MAD_FLOOR)
    sign = np.zeros(len(x), dtype=int)   # 0 = within threshold or missing
    sign[(x - med) > thr] = 1            # above median
    sign[(x - med) < -thr] = -1          # below median
    dat_xr_pt[f'{idx}_clim_med'] = ('time', med)
    dat_xr_pt[f'{idx}_thr'] = ('time', thr)
    dat_xr_pt[f'{idx}_sign'] = ('time', sign)


MIN_RUN = 3   # "more than 2 in a row"

def run_lengths(sign):
    """Length of the consecutive same-sign run each flagged step belongs to. Missing (0) breaks a run."""
    run_len = np.zeros(len(sign), dtype=int)
    t = 0
    while t < len(sign):
        if sign[t] == 0:
            t += 1
            continue
        start = t
        while t + 1 < len(sign) and sign[t + 1] == sign[start]:
            t += 1
        run_len[start:t + 1] = t - start + 1
        t += 1
    return run_len

for idx in CLIM_VARS:
    sign = dat_xr_pt[f'{idx}_sign'].values
    run_len = run_lengths(sign)
    flag = np.zeros(len(sign), dtype=int)           # 0 = within threshold
    flag[(sign != 0) & (run_len < MIN_RUN)] = 1     # transient outlier -> remove
    flag[(sign != 0) & (run_len >= MIN_RUN)] = 2    # persistent anomaly -> keep
    dat_xr_pt[f'{idx}_run'] = ('time', run_len)
    dat_xr_pt[f'{idx}_flag'] = ('time', flag)

# outlier if transient in either index, target year only (padding steps just serve as neighbors)
outlier = ((dat_xr_pt['NDVI_flag'].values == 1) | (dat_xr_pt['CCI_flag'].values == 1)) & is_target
dat_xr_pt['outlier'] = ('time', outlier)


#%%
#########################################
# Step 6: inspect
#########################################
cols = ['NDVI', 'NDVI_clim_med', 'NDVI_sign', 'NDVI_run', 'NDVI_flag',
        'CCI', 'CCI_clim_med', 'CCI_sign', 'CCI_run', 'CCI_flag', 'outlier']
df = dat_xr_pt[cols].to_dataframe()[cols]
df.insert(0, 'cp', step_cps)
print(df.round(3).to_string())

flag_colors = {0: 'black', 1: 'red', 2: 'royalblue'}
flag_labels = {0: 'within threshold', 1: 'transient outlier', 2: 'persistent'}
t = dat_xr_pt['time'].values

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
for ax, idx in zip(axes, ['NDVI', 'CCI']):
    x = dat_xr_pt[idx].values
    med = dat_xr_pt[f'{idx}_clim_med'].values
    thr = dat_xr_pt[f'{idx}_thr'].values
    flag = dat_xr_pt[f'{idx}_flag'].values
    ax.fill_between(t, med - thr, med + thr, color='gray', alpha=0.25, label=f'median ± {K:g}×MAD')
    ax.plot(t, med, color='gray', lw=1)
    for code in (0, 1, 2):
        m = flag == code
        ax.scatter(t[m], x[m], color=flag_colors[code], s=25, zorder=3, label=flag_labels[code])
    ax.scatter(t[~is_target], x[~is_target], facecolor='none', edgecolor='orange', s=60, zorder=4,
               label='padding (not flagged)')
    ax.set_ylabel(idx)
axes[0].legend(fontsize=8)
axes[0].set_title(f'{site} {composite_year}')
plt.tight_layout()
plt.show()


