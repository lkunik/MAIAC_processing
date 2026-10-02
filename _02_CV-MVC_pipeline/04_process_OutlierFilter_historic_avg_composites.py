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
def main():
    try:
        # mark start time to keep track of elapsed
        start_total = time.time()


        MODIS_tile = sys.argv[1] #'h10v04'
        composite_period = int(sys.argv[2]) # command line argument
        QC_option = sys.argv[3] # command line argument - options are 'CloudFree', 'CloudAOD', 'Cloud', 'AOD', or 'None'

        dat_dir = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/pre-fill_OutlierFilter/QCfilt_{QC_option}/{MODIS_tile}/'
        plot_dir = f'/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/plots/maps/MAIAC_CV-MVC/1km_curv/hist_avg/QCfilt_{QC_option}/{MODIS_tile}/'
        filtered_out_dir = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/pre-fill_OutlierFilter/QCfilt_{QC_option}/{MODIS_tile}/'
        out_dir = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/pre-fill_OutlierFilter/QCfilt_{QC_option}/{MODIS_tile}/hist_avg/'
        Path(plot_dir).mkdir(parents=True, exist_ok=True)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        Path(filtered_out_dir).mkdir(parents=True, exist_ok=True)

        min_valid_count = 3  # minimum number of valid points required to compute mean for a composite period
        composite_periods = np.arange(0, 23)  # 0-22
        # NDVI_arr_dict_all = {cp: {} for cp in composite_periods}
        # NDVI_arr_dict_filter = {cp: {} for cp in composite_periods}
        # CCI_arr_dict_all = {cp: {} for cp in composite_periods}
        # CCI_arr_dict_filter = {cp: {} for cp in composite_periods}

        outfile = os.path.join(out_dir, f'MCD19A1.{MODIS_tile}.061_1km_CV-MVC_QCfilt_{QC_option}_hist_avg_{composite_period:02d}.nc')
        # if os.path.exists(outfile):
        #     print(f'FYI - Output file already exists, but continuing anayways....')
        #     sys.exit(0)

        # Skip processing if 'v01' is in MODIS_tile and composite_period is in [0, 1, 2, 3, 19, 20, 21, 22]
        if 'v01' in MODIS_tile and composite_period in [0, 1, 2, 3, 19, 20, 21, 22]:
            print(f"Skipping composite_period {composite_period} for MODIS_tile {MODIS_tile} (high latitude winter, no data).")
            sys.exit(0)

        # Skip processing if 'v02' is in MODIS_tile and composite_period is in [0, 1, 20, 21, 22]
        if 'v02' in MODIS_tile and composite_period in [0, 1, 20, 21, 22]:
            print(f"Skipping composite_period {composite_period} for MODIS_tile {MODIS_tile} (high latitude winter, no data).")
            sys.exit(0)

        # List all files for this month portion
        MCD19A1_files = sorted(glob.glob(f'{dat_dir}MCD19A1*-{composite_period:02d}.nc'))
        if len(MCD19A1_files) == 0:
            print(f'No files found for composite_period {composite_period}, should be more than 0 files, returning error code')
            sys.exit(1)

        dat_xrs = []
        for file in MCD19A1_files:
            try:
                print(f'Loading file: {file}')
                dat_ds = xr.open_dataset(file)
                if 'NDVI_rank' in dat_ds:
                    dat_ds = dat_ds.drop_vars(['NDVI_rank'])
                if 'VZA_rank' in dat_ds:
                    dat_ds = dat_ds.drop_vars(['VZA_rank'])
                if 'QA_flag' in dat_ds:
                    dat_ds = dat_ds.drop_vars(['QA_flag'])  # categorical flag, not meaningful to average
                dat_ds = dat_ds.drop_vars(['orbit_time_utc', 'orbit_local_solar_time'], errors='ignore')  # optional per-pixel orbit time diagnostic from step 01, not meaningful to average
                dat_ds = dat_ds.assign(NDVI=(dat_ds['Sur_refl2'] - dat_ds['Sur_refl1']) / (dat_ds['Sur_refl2'] + dat_ds['Sur_refl1']))
                dat_ds = dat_ds.assign(CCI=(dat_ds['Sur_refl11'] - dat_ds['Sur_refl1']) / (dat_ds['Sur_refl11'] + dat_ds['Sur_refl1']))
                dat_xrs.append(dat_ds)
            except Exception as e:
                print(f'Error loading file {file}: {e}, skipping...')

        combined_ds_all = xr.concat(dat_xrs, dim='time')
        #----------------------------------------
        # count valid years per pixel, and blank out pixels with too few before computing any statistics
        #----------------------------------------
        MIN_VALID_YEARS = 4   # minimum number of years with valid NDVI required to compute climatology statistics (3 or fewer -> all statistics NaN)

        ndvi_valid_count = combined_ds_all['NDVI'].count(dim='time')   # 2D (y, x): number of years with a valid (non-NaN) NDVI at each pixel, counted before masking
        sufficient = ndvi_valid_count >= MIN_VALID_YEARS               # 2D boolean (y, x): True where there are enough years to compute statistics

        for v in combined_ds_all.data_vars:                            # loop over every variable in the stacked dataset (bands, geometry, NDVI, CCI)
            if ('y' in combined_ds_all[v].dims) and ('x' in combined_ds_all[v].dims):   # only per-pixel variables (variables without (y, x) are left untouched)
                combined_ds_all[v] = combined_ds_all[v].where(sufficient)                # set every year to NaN at pixels with fewer than MIN_VALID_YEARS valid years

        n_few = int(((ndvi_valid_count > 0) & ~sufficient).sum())     # number of pixels that had some data (1-3 years) but not enough
        print(f'{n_few} pixels with 1-{MIN_VALID_YEARS - 1} valid years set to NaN for composite period {composite_period:02d}')

        #----------------------------------------
        # climatology statistics (NaN wherever there were too few valid years)
        #----------------------------------------
        with warnings.catch_warnings():                                # all-NaN pixels trigger "Mean of empty slice" / "All-NaN slice" warnings;
            warnings.simplefilter('ignore', category=RuntimeWarning)   # the result there is NaN, which is what we want, so the warnings are suppressed

            average_ds = combined_ds_all.mean(dim='time', skipna=True)     # historical mean of every variable, per pixel (saved as NDVI, CCI, Sur_refl*, ...)
            median_ds = combined_ds_all.median(dim='time', skipna=True)    # historical median of every variable, per pixel

            # median absolute deviation: median over years of |value - median| (raw, unscaled)
            ndvi_mad = (np.abs(combined_ds_all['NDVI'] - median_ds['NDVI'])).median(dim='time', skipna=True)
            cci_mad = (np.abs(combined_ds_all['CCI'] - median_ds['CCI'])).median(dim='time', skipna=True)

            std_ds = combined_ds_all.std(dim='time', skipna=True)          # historical standard deviation of every variable, per pixel

        average_ds['NDVI_median'] = median_ds['NDVI']        # historical median NDVI (this is what the outlier test should compare against)
        average_ds['CCI_median'] = median_ds['CCI']          # historical median CCI
        average_ds['NDVI_MAD'] = ndvi_mad                    # MAD of NDVI
        average_ds['CCI_MAD'] = cci_mad                      # MAD of CCI
        average_ds['NDVI_std'] = std_ds['NDVI']              # standard deviation of NDVI
        average_ds['CCI_std'] = std_ds['CCI']                # standard deviation of CCI
        average_ds['NDVI_valid_count'] = ndvi_valid_count    # number of valid years per pixel (unmasked, so downstream steps can check it)

        #----------------------------------------
        # flag pixels with anomalously high std, judged against the regular (pre-fill) hist_avg climatology for this tile
        #----------------------------------------
        # A pixel is flagged (all statistics set to NaN, like pixels below MIN_VALID_YEARS) if EITHER NDVI OR CCI has
        #   std > range (max - min) of that pixel's pre-fill hist_avg median across all composite periods
        ref_hist_avg_dir = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/pre-fill/QCfilt_{QC_option}/{MODIS_tile}/hist_avg/'
        ref_hist_avg_files = sorted(glob.glob(f'{ref_hist_avg_dir}MCD19A1.{MODIS_tile}.*_hist_avg_??.nc'))
        if len(ref_hist_avg_files) == 0:
            print(f'No pre-fill hist_avg files found in {ref_hist_avg_dir} (run 02_process_historic_avg_composites.py first), returning error code')
            sys.exit(1)
        print(f'Computing std thresholds from {len(ref_hist_avg_files)} pre-fill hist_avg files in {ref_hist_avg_dir}')

        ref_vars = ['NDVI_median', 'CCI_median']
        ref_stacks = {v: [] for v in ref_vars}                             # each becomes a 3D array (composite period, y, x)
        for ref_file in ref_hist_avg_files:
            with xr.open_dataset(ref_file) as ref_ds:
                for v in ref_vars:
                    ref_stacks[v].append(ref_ds[v].transpose('y', 'x').values)
        ref_stacks = {v: np.stack(arrs) for v, arrs in ref_stacks.items()}

        high_std_flag = np.zeros(average_ds['NDVI_std'].transpose('y', 'x').shape, dtype=bool)
        with warnings.catch_warnings():                                # all-NaN pixels (e.g. water) trigger "All-NaN slice" warnings; thresholds there are NaN,
            warnings.simplefilter('ignore', category=RuntimeWarning)   # and comparisons against NaN are False, so those pixels are never flagged
            for var_name in ['NDVI', 'CCI']:
                median_range = np.nanmax(ref_stacks[f'{var_name}_median'], axis=0) - np.nanmin(ref_stacks[f'{var_name}_median'], axis=0)   # 2D (y, x)
                new_std = average_ds[f'{var_name}_std'].transpose('y', 'x').values

                exceeds_range = new_std > median_range
                high_std_flag |= exceeds_range
                print(f'{var_name}: {int(exceeds_range.sum())} pixels with std > range of pre-fill hist_avg median')
        del ref_stacks

        high_std_flag = xr.DataArray(high_std_flag, dims=('y', 'x'), coords={'y': average_ds['y'], 'x': average_ds['x']})
        for v in average_ds.data_vars:
            if v == 'NDVI_valid_count':                                # kept unmasked, as for the MIN_VALID_YEARS check
                continue
            if ('y' in average_ds[v].dims) and ('x' in average_ds[v].dims):
                average_ds[v] = average_ds[v].where(~high_std_flag)
        average_ds['high_std_flag'] = high_std_flag.astype('uint8')    # 1 where statistics were removed by the high-std check

        print(f'{int(high_std_flag.sum())} pixels flagged for high NDVI or CCI std and set to NaN for composite period {composite_period:02d}')

        # Finally, save to netcdf
        average_ds.to_netcdf(path=outfile, format="NETCDF4")
        average_ds.close()

        # # APPLY OUTLIER DETECTION (no flattening)
        # outlier_mask, stats = detect_outliers_mad(
        #     combined_ds_all['NDVI'].values,  # keep original shape (time, y, x)
        #     combined_ds_all['CCI'].values,
        #     k=3,
        #     min_valid_count=4
        # )
        
        # combined_ds_filter = combined_ds_all.copy()
        # # Remove outliers from dataset
        # if np.any(outlier_mask):
        #     combined_ds_filter = combined_ds_all.where(~outlier_mask, np.nan) 

        # for tt in combined_ds_filter.time:
        #     dat_xr = combined_ds_filter.sel(time=tt)
        #     composite_year = int(dat_xr.time.dt.year.values)
        #     outfile_filtered = os.path.join(filtered_out_dir, f'MCD19A1.{MODIS_tile}.061_1km_16day_CV-MVC_QCfilt_{QC_option}_OutlierFilter_{composite_year}-{str(composite_period).zfill(2)}.nc')
        #     dat_xr.to_netcdf(path=outfile_filtered, format="NETCDF4")
        #     dat_xr.close()
        #     print(f'saved outlier-filtered file to {outfile_filtered}')

        # # take the time-wise average ignoring NaNs
        # average_ds = combined_ds_filter.mean(dim='time', skipna=True)
        # std_ds = combined_ds_filter.std(dim='time', skipna=True)

        # # isolate NDVI and count non-NaN per pixel across time
        # ndvi_stack = combined_ds_filter['NDVI']  # DataArray with dim 'time', 'lat', 'lon'
        # ndvi_valid_count = ndvi_stack.count(dim='time')  # counts non-NA values per pixel
        # average_ds['NDVI_valid_count'] = ndvi_valid_count
        # average_ds['NDVI_std'] = std_ds['NDVI']
        # average_ds['CCI_std'] = std_ds['CCI']


        # del combined_ds_all, combined_ds_filter, ndvi_stack, ndvi_valid_count # remove large variables from environment to free up space
        # gc.collect()

        # # # Count NaN pixels in average_ds.NDVI
        # # nan_count = int(average_ds.NDVI.isnull().sum().values)  # total number of NaNs
        # # total_pixels = int(average_ds.cosVZA.size)  # total number of pixels
        # # nan_pct = 100.0 * nan_count / total_pixels if total_pixels > 0 else np.nan
        # # print(f'NDVI NaN pixels: {nan_count} / {total_pixels} ({nan_pct:.2f}%)')

        # # Finally, save to netcdf
        # average_ds.to_netcdf(path=outfile, format="NETCDF4")
        # average_ds.close()

        # # plot_ds_NDVI = average_ds.NDVI
        # # plot_ds_NDVI_valid_count = average_ds.NDVI_valid_count

        # del average_ds # remove large variables from environment to free up space
        # gc.collect()


        # plotfile = os.path.join(plot_dir, f'MCD19A1.{MODIS_tile}.061_1km_CV-MVC_QCfilt_{QC_option}_hist_avg_{composite_period:02d}_NDVI.png')
        # test_plot(dat_da = plot_ds_NDVI, 
        #     plot_title=f'Period {composite_period:02d}', zlab='NDVI',
        #     plot_filename=plotfile,
        #     plot_min = 0, plot_max = 1.2)

        # plotfile = os.path.join(plot_dir, f'MCD19A1.{MODIS_tile}.061_1km_CV-MVC_QCfilt_{QC_option}_hist_avg_{composite_period:02d}_NDVI_valid_count.png')
        # test_plot(dat_da = plot_ds_NDVI_valid_count, 
        #     plot_title=f'Period {composite_period:02d}', zlab='NDVI Valid Count',
        #     plot_filename=plotfile,
        #     plot_min = 0, plot_max = 25)

        # del plot_ds_NDVI, plot_ds_NDVI_valid_count # remove large variables from environment to free up space
        # gc.collect()

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
