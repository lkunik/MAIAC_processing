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
        'tile': 'h09v04',
        'coords': (-105.5464, 40.0329),
        'start': '2017-01-01',
        'end': '2018-12-31'
    },
    'OSBS': {
        'tile': 'h10v06',
        'coords': (-81.993431, 29.689282),
        'start': '2021-01-01',
        'end': '2022-12-31'
    },
    'Ca-Obs': {
        'tile': 'h11v03',
        'coords': (-105.1178, 53.9872),
        'start': '2018-01-01',
        'end': '2022-12-31'
    },
    'DEJU': {
        'tile': 'h11v02',
        'coords': (-145.75136, 63.88112),
        'start': '2019-01-01',
        'end': '2021-12-31'
    }
}

#########################################
# Begin main
#########################################
#%%

# mark start time to keep track of elapsed
start_total = time.time()



MODIS_tile = 'h10v06' # sys.argv[1] #'h10v04'
composite_period = 14 # int(sys.argv[2]) # command line argument
QC_option = 'CloudFree_LowAOD_ClearAdj' #sys.argv[3] 

dat_dir = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/pre-fill/QCfilt_{QC_option}/{MODIS_tile}/'
plot_dir = f'/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/plots/maps/MAIAC_CV-MVC/1km_curv/hist_avg/QCfilt_{QC_option}/{MODIS_tile}/'
filtered_out_dir = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/pre-fill_OutlierFilter/QCfilt_{QC_option}/{MODIS_tile}/'
out_dir = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/pre-fill/QCfilt_{QC_option}/{MODIS_tile}/hist_median/'
Path(plot_dir).mkdir(parents=True, exist_ok=True)
Path(out_dir).mkdir(parents=True, exist_ok=True)
Path(filtered_out_dir).mkdir(parents=True, exist_ok=True)

min_valid_count = 3  # minimum number of valid points required to compute mean for a composite period
composite_periods = np.arange(0, 23)  # 0-22
# NDVI_arr_dict_all = {cp: {} for cp in composite_periods}
# NDVI_arr_dict_filter = {cp: {} for cp in composite_periods}
# CCI_arr_dict_all = {cp: {} for cp in composite_periods}
# CCI_arr_dict_filter = {cp: {} for cp in composite_periods}


outfile = os.path.join(out_dir, f'MCD19A1.{MODIS_tile}.061_1km_CV-MVC_QCfilt_{QC_option}_hist_median_{composite_period:02d}.nc')
# if os.path.exists(outfile):
#     print(f'FYI - Output file already exists, but continuing anayways....')
#     sys.exit(0)

# Skip processing if 'v01' is in MODIS_tile and composite_period is in [0, 1, 2, 3, 19, 20, 21, 22]
if 'v01' in MODIS_tile and composite_period in [0, 1, 2, 3, 19, 20, 21, 22]:
    print(f"Skipping composite_period {composite_period} for MODIS_tile {MODIS_tile} (high latitude winter, no data).")
    # sys.exit(0)

# Skip processing if 'v02' is in MODIS_tile and composite_period is in [0, 1, 20, 21, 22]
if 'v02' in MODIS_tile and composite_period in [0, 1, 20, 21, 22]:
    print(f"Skipping composite_period {composite_period} for MODIS_tile {MODIS_tile} (high latitude winter, no data).")
    # sys.exit(0)

# List all files for this month portion
MCD19A1_files = sorted(glob.glob(f'{dat_dir}MCD19A1*-{composite_period:02d}.nc'))
if len(MCD19A1_files) == 0:
    print(f'No files found for composite_period {composite_period}, should be more than 0 files, returning error code')
    # sys.exit(1)

dat_xrs = []
for file in MCD19A1_files:
    try:
        print(f'Loading file: {file}')
        dat_ds = xr.open_dataset(file)
        if 'NDVI_rank' in dat_ds:
            dat_ds = dat_ds.drop_vars(['NDVI_rank'])
        if 'VZA_rank' in dat_ds:
            dat_ds = dat_ds.drop_vars(['VZA_rank'])
        dat_ds = dat_ds.assign(NDVI=(dat_ds['Sur_refl2'] - dat_ds['Sur_refl1']) / (dat_ds['Sur_refl2'] + dat_ds['Sur_refl1']))
        dat_ds = dat_ds.assign(CCI=(dat_ds['Sur_refl11'] - dat_ds['Sur_refl1']) / (dat_ds['Sur_refl11'] + dat_ds['Sur_refl1']))
        dat_xrs.append(dat_ds)
    except Exception as e:
        print(f'Error loading file {file}: {e}, skipping...')

combined_ds_all = xr.concat(dat_xrs, dim='time')

median_ds = combined_ds_all.median(dim='time', skipna=True)

# Compute Median Absolute Deviation (MAD) for NDVI and CCI
ndvi_mad = (np.abs(combined_ds_all['NDVI'] - median_ds['NDVI'])).median(dim='time', skipna=True)
cci_mad = (np.abs(combined_ds_all['CCI'] - median_ds['CCI'])).median(dim='time', skipna=True)
median_ds['NDVI_MAD'] = ndvi_mad
median_ds['CCI_MAD'] = cci_mad



std_ds = combined_ds_all.std(dim='time', skipna=True)
ndvi_stack = combined_ds_all['NDVI']  # DataArray with dim 'time', 'lat', 'lon'
ndvi_valid_count = ndvi_stack.count(dim='time')  # counts non-NA values per pixel
median_ds['NDVI_valid_count'] = ndvi_valid_count
median_ds['NDVI_std'] = std_ds['NDVI']
median_ds['CCI_std'] = std_ds['CCI']
# Finally, save to netcdf
median_ds.to_netcdf(path=outfile, format="NETCDF4")
median_ds.close()

end_total = time.time()
total_elapsed = timedelta(seconds=end_total-start_total)
print(f'File saved. Total elapsed time = {td_min(total_elapsed)} mins, {td_sec(total_elapsed)} seconds')



# %%
