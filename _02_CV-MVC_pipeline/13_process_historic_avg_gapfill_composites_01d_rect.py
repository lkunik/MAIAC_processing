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

# Custom packages
utils_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/pyutil'
if utils_dir not in sys.path:
    sys.path.append(utils_dir)
from MODIS_tile_corners import MODIS_tilecorners_df

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



#########################################
# Define Global Variables and constants
#########################################

hist_years = np.arange(2000, 2025)
DOY_composite_starts = np.arange(1, 366, 16)
hist_composites = range(len(DOY_composite_starts))  # 0-22

#########################################
# Begin main
#########################################
#%%
def main():
    try:
        # mark start time to keep track of elapsed
        start_total = time.time()


        MODIS_tile = sys.argv[1] #'h10v04'
        composite_period = int(sys.argv[2]) # command line argument
        QC_option = sys.argv[3] # command line argument - options are 'CloudFree', 'CloudAOD', 'Cloud', 'AOD', or 'None'

        dat_dir = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/01d_rect/QCfilt_{QC_option}/{MODIS_tile}/'
        plot_dir = f'/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/plots/maps/MAIAC_CV-MVC/01d_rect/hist_avg_gapfill/QCfilt_{QC_option}/{MODIS_tile}/'
        out_dir = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/01d_rect/QCfilt_{QC_option}/{MODIS_tile}/hist_avg/'
        Path(plot_dir).mkdir(parents=True, exist_ok=True)
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        # MCD19A1.h12v02.061_01d_16day_CV-MVC_QCfilt_CloudFree_2025-19_rect.nc
        outfile = os.path.join(out_dir, f'MCD19A1.{MODIS_tile}.061_01d_CV-MVC_QCfilt_{QC_option}_hist_avg_{composite_period:02d}.nc')
        if os.path.exists(outfile):
            print(f'Output file already exists, skipping: {outfile}')
            # sys.exit(0)


        # Skip processing if 'v02' is in MODIS_tile and composite_period is in [0, 1, 20, 21, 22]
        if 'v02' in MODIS_tile and composite_period in [0, 1, 20, 21, 22]:
            print(f"Skipping composite_period {composite_period} for MODIS_tile {MODIS_tile} (high latitude winter, no data).")
            sys.exit(0)

        # List all files for this month portion
        MCD19A1_files = sorted(glob.glob(f'{dat_dir}MCD19A1*-{composite_period:02d}_rect.nc'))
        if len(MCD19A1_files) == 0:
            print(f'No files found for composite_period {composite_period}, should be more than 0 files, returning error code')
            sys.exit(1)


        dat_xrs = []
        for file in MCD19A1_files:
            yyyy = int(os.path.basename(file).split('.')[-2].split('_')[-2].split('-')[0])
            try:
                print(f'Loading file: {file}')
                dat_ds = xr.open_dataset(file)
                if 'NDVI_rank' in dat_ds:
                    dat_ds = dat_ds.drop_vars(['NDVI_rank'])
                if 'VZA_rank' in dat_ds:
                    dat_ds = dat_ds.drop_vars(['VZA_rank'])
                if 'QA_flag' in dat_ds:
                    dat_ds = dat_ds.drop_vars(['QA_flag'])  # categorical flag, not meaningful to average
                dat_ds.rio.write_crs("EPSG:4326", inplace=True) # ensure the dataset has a CRS assigned for later reprojection if needed
                dat_xrs.append(dat_ds)
            except Exception as e:
                print(f'Error loading file {file}: {e}, skipping...')
        combined_ds = xr.concat(dat_xrs, dim='time', coords='minimal')

        combined_ds = combined_ds.assign(CCI=(combined_ds["Sur_refl11"] - combined_ds["Sur_refl1"])/
                    (combined_ds["Sur_refl11"] + combined_ds["Sur_refl1"]))
        # combined_ds = combined_ds.assign(NDMI=(combined_ds["Sur_refl6"] - combined_ds["Sur_refl1"])/
        #             (combined_ds["Sur_refl6"] + combined_ds["Sur_refl1"]))
        combined_ds = combined_ds.assign(NDVI=(combined_ds["Sur_refl2"] - combined_ds["Sur_refl1"])/
                    (combined_ds["Sur_refl2"] + combined_ds["Sur_refl1"]))
        # combined_ds = combined_ds.assign(NDSI=(combined_ds["Sur_refl4"] - combined_ds["Sur_refl6"])/
        #             (combined_ds["Sur_refl4"] + combined_ds["Sur_refl6"]))
        # combined_ds = combined_ds.assign(NIRv=(combined_ds["Sur_refl2"] - combined_ds["Sur_refl1"])*combined_ds["Sur_refl2"]/
        #             (combined_ds["Sur_refl2"] + combined_ds["Sur_refl1"]))

        # take the time-wise average ignoring NaNs
        average_ds = combined_ds.mean(dim='time', skipna=True)
        average_ds['CCI_std'] = combined_ds.CCI.std(dim='time', skipna=True)
        average_ds['NDVI_std'] = combined_ds.NDVI.std(dim='time', skipna=True)
        # average_ds['NDMI_std'] = combined_ds.NDMI.std(dim='time', skipna=True)
        # average_ds['NDSI_std'] = combined_ds.NDSI.std(dim='time', skipna=True)
        # average_ds['NIRv_std'] = combined_ds.NIRv.std(dim='time', skipna=True)

        # isolate NDVI and count non-NaN per pixel across time
        ndvi_stack = combined_ds['NDVI']  # DataArray with dim 'time', 'lat', 'lon'
        ndvi_valid_count = ndvi_stack.count(dim='time')  # counts non-NA values per pixel
        average_ds['NDVI_valid_count'] = ndvi_valid_count
        average_ds.rio.write_crs("EPSG:4326", inplace=True) # ensure the dataset has a CRS assigned for later reprojection if needed

        del combined_ds, ndvi_stack, ndvi_valid_count # remove large variables from environment to free up space
        gc.collect()

        # # Count NaN pixels in average_ds.NDVI
        # nan_count = int(average_ds.NDVI.isnull().sum().values)  # total number of NaNs
        # total_pixels = int(average_ds.cosVZA.size)  # total number of pixels
        # nan_pct = 100.0 * nan_count / total_pixels if total_pixels > 0 else np.nan
        # print(f'NDVI NaN pixels: {nan_count} / {total_pixels} ({nan_pct:.2f}%)')

        # Finally, save to netcdf
        average_ds.to_netcdf(path=outfile, format="NETCDF4")
        average_ds.close()

        plot_ds_NDVI = average_ds.NDVI
        plot_ds_NDVI_std = average_ds.NDVI_std
        plot_ds_CCI = average_ds.CCI
        plot_ds_CCI_std = average_ds.CCI_std
        plot_ds_NDVI_valid_count = average_ds.NDVI_valid_count

        del average_ds # remove large variables from environment to free up space
        gc.collect()


        plotfile = os.path.join(plot_dir, f'MCD19A1.{MODIS_tile}.061_1km_CV-MVC_lmFill_QCfilt_{QC_option}_hist_avg_{composite_period:02d}_CCI.png')
        test_plot(dat_da = plot_ds_CCI, 
            plot_title=f'Period {composite_period:02d}', zlab='CCI',
            plot_filename=plotfile,
            plot_min = -0.36, plot_max = 0.3)

        plotfile = os.path.join(plot_dir, f'MCD19A1.{MODIS_tile}.061_1km_CV-MVC_lmFill_QCfilt_{QC_option}_hist_avg_{composite_period:02d}_CCI_std.png')
        test_plot(dat_da = plot_ds_CCI_std, 
            plot_title=f'Period {composite_period:02d}', zlab='CCI Std',
            plot_filename=plotfile,
            plot_min = 0, plot_max = 0.2)
        
        del plot_ds_NDVI, plot_ds_CCI, plot_ds_NDVI_valid_count # remove large variables from environment to free up space
        gc.collect()

        end_total = time.time()
        total_elapsed = timedelta(seconds=end_total-start_total)
        print(f'File saved. Total elapsed time = {td_min(total_elapsed)} mins, {td_sec(total_elapsed)} seconds')

    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)  

if __name__ == '__main__':
   main()
# %%
