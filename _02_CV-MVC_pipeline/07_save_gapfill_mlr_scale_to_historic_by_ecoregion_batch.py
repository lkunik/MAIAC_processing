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


EPA_ecoregion_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/EPA/ecoregions/L2/NAm_shp_FOR_MAIAC_PROC/'
EPA_ecoregion_L2_file = os.path.join(EPA_ecoregion_dir, 'NA_CEC_Eco_Level2_withOther.shp')
EPA_ecoregion_regrid_mask_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/EPA/ecoregions/L2/for_MAIAC_proc/NAm/'

# EPA_ecoregion_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/EPA/ecoregions/L3/'
# EPA_ecoregion_L3_file = os.path.join(EPA_ecoregion_dir, 'NAm_cec_eco_l3.shp')
# EPA_ecoregion_regrid_mask_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/EPA/ecoregions/L3/for_MAIAC_proc/NAm/'

LC_basedir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/ESA_CCI/LandCover/for_MAIAC_proc/'
forest_frac_dir = os.path.join(LC_basedir, 'forest_fraction/')
crop_frac_dir = os.path.join(LC_basedir, 'crop_fraction/')
otherveg_frac_dir = os.path.join(LC_basedir, 'other_vegetation_fraction/')

elev_regrid_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/USGS/GTOPO30/for_MAIAC_proc/'

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
        plot_min = float(np.nanmin(dat_da))
    if plot_max is None:
        plot_max = float(np.nanmax(dat_da))
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


composite_years = np.arange(2000, 2025)
DOY_composite_starts = np.arange(1, 366, 16)
composite_periods = range(len(DOY_composite_starts))  # 0-22

refl_vars = ['Sur_refl1', 'Sur_refl2', 'Sur_refl11', 'Sigma_BRFn1', 'Sigma_BRFn2']
#########################################
# Begin main
#########################################
#%%
def main():

    try:
        # mark start time to keep track of elapsed
        start_total = time.time()

        MODIS_tile = sys.argv[1] #'h10v04'
        composite_year = int(sys.argv[2]) # command line argument
        composite_period = int(sys.argv[3]) # command line argument
        QC_option = sys.argv[4] # command line argument - options are 'CloudFree', 'CloudAOD', 'Cloud', 'AOD', or 'None'

        # dat_dir = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/pre-fill/QCfilt_{QC_option}/{MODIS_tile}/'
        dat_dir = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/pre-fill_OutlierFilter/QCfilt_{QC_option}/{MODIS_tile}/'
        MCD19A1_files = sorted(glob.glob(os.path.join(dat_dir, f'MCD19A1.{MODIS_tile}.*.nc')))

        hist_dir = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/pre-fill_OutlierFilter/QCfilt_{QC_option}/{MODIS_tile}/hist_avg'
        MCD19A1_hist_files = sorted(glob.glob(os.path.join(hist_dir, f'MCD19A1.{MODIS_tile}.*.nc')))
        out_dir = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/gapfill_scale/QCfilt_{QC_option}/'
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        forest_frac_file = os.path.join(forest_frac_dir, f'forest_frac_1km_curv_{MODIS_tile}.nc')
        crop_frac_file = os.path.join(crop_frac_dir, f'crop_frac_1km_curv_{MODIS_tile}.nc')
        otherveg_frac_file = os.path.join(otherveg_frac_dir, f'otherveg_frac_1km_curv_{MODIS_tile}.nc')
        elev_regrid_file = os.path.join(elev_regrid_dir, f'GTOPO30_DEM_1km_curv_{MODIS_tile}.nc') 


        # Load in Land cover fraction datasets, pre-gridded to 1km curvilinear grid
        forest_frac_xr_curv = xr.open_dataset(forest_frac_file)['__xarray_dataarray_variable__']
        crop_frac_xr_curv = xr.open_dataset(crop_frac_file)['__xarray_dataarray_variable__']
        otherveg_frac_xr_curv = xr.open_dataset(otherveg_frac_file)['__xarray_dataarray_variable__']

        # Load in Elevation dataset and regrid to 1km curvilinear grid
        elev_xr_curv = xr.open_dataset(elev_regrid_file)['__xarray_dataarray_variable__']
        # test_plot(elev_xr_curv, 'Elevation 1km Curvilinear Grid', 'Elevation (m)', plot_filename=None, plot_min=0, plot_max=4000)

        ecoregion_L2_gdf = gpd.read_file(EPA_ecoregion_L2_file)
        NAm_ecoregions = np.unique(ecoregion_L2_gdf.NA_L2NAME)

        # Skip processing if 'v01' is in MODIS_tile and composite_period is in [0, 1, 2, 3, 19, 20, 21, 22]
        if 'v01' in MODIS_tile and composite_period in [0, 1, 2, 3, 19, 20, 21, 22]:
            print(f"Skipping composite_period {composite_period} for MODIS_tile {MODIS_tile} (high latitude winter, no data).")
            sys.exit(0)
            
        # Skip processing if 'v02' is in MODIS_tile and composite_period is in [0, 1, 20, 21, 22]
        if 'v02' in MODIS_tile and composite_period in [0, 1, 20, 21, 22]:
            print(f"Skipping composite_period {composite_period} for MODIS_tile {MODIS_tile} (high latitude winter, no data).")
            sys.exit(0)

        hist_files = [f for f in MCD19A1_hist_files if f.endswith(f'{composite_period:02d}.nc')]
        if len(hist_files) == 0:
            print(f'Historical average period does not exist for tile {MODIS_tile} period {composite_period}, exiting with error code...')
            sys.exit(1)
        else:
            hist_file = hist_files[0]
        hist_ds = xr.open_dataset(hist_file)[refl_vars]

        # Find the file that matches the current composite_year and composite_period
        dat_files = [f for f in MCD19A1_files if f'{composite_year}-{composite_period:02d}.nc' in f]
        if len(dat_files) == 0:
            print(f'No data file found for year {composite_year}, period {composite_period}, exiting with error code...')
            sys.exit(0)
        else:
            dat_file = dat_files[0]
        
        
        dat_ds_temp = xr.open_dataset(dat_file)
        if 'time' in dat_ds_temp.dims:
            dat_ds = dat_ds_temp.squeeze('time')[refl_vars]
        else:
            dat_ds = dat_ds_temp[refl_vars]

        print(f'Gap-filling with climatology - composite year: {composite_year}, time period {composite_period}')
        # represent the scaling from historical to this year's data as a ratio
        scale_ds_raw = dat_ds / hist_ds

        # Loop through the ecoregions and create masks
        for NAm_ecoregion in NAm_ecoregions:

            csv_output_file = os.path.join(out_dir, f'{MODIS_tile}_scale_factors_{NAm_ecoregion.replace(" ", "_").replace("/", "_")}_{composite_year}-{composite_period:02d}.csv')
            # if os.path.exists(csv_output_file):
            #     print(f'Scaling factors CSV for {NAm_ecoregion} already exists, skipping...')
            #     continue

            #"EPA_L2_ecoregion_mask_1km_curv_h09v04_Arizona_New_Mexico_Mountains.nc"
            ecoregion_mask_file = os.path.join(EPA_ecoregion_regrid_mask_dir, f'EPA_L2_ecoregion_mask_1km_curv_{MODIS_tile}_{NAm_ecoregion.replace(" ", "_").replace("/", "_")}.nc')
            if not os.path.exists(ecoregion_mask_file):
                print(f'Ecoregion mask file {ecoregion_mask_file} does not exist, skipping...')
                continue
            xr_ecoregion_mask_curv = xr.open_dataset(ecoregion_mask_file)['__xarray_dataarray_variable__']
            if (xr_ecoregion_mask_curv == 0).all():
                print(f'Ecoregion mask is all zeros for {NAm_ecoregion}, skipping...')
                continue
            
            print(f'Ecoregion masking for {NAm_ecoregion}')
            scale_ds_ecoregion_clip = scale_ds_raw.where(xr_ecoregion_mask_curv == 1)
            forest_frac_ecoregion_clip = forest_frac_xr_curv.where(xr_ecoregion_mask_curv == 1)
            crop_frac_ecoregion_clip = crop_frac_xr_curv.where(xr_ecoregion_mask_curv == 1)
            otherveg_frac_ecoregion_clip = otherveg_frac_xr_curv.where(xr_ecoregion_mask_curv == 1)
            elev_ecoregion_clip = elev_xr_curv.where(xr_ecoregion_mask_curv == 1)

            # Flatten the data arrays and create a mask for valid (non-NaN) entries
            valid_mask = np.full_like(scale_ds_ecoregion_clip[refl_vars[0]], False, dtype=bool)
            for var in refl_vars:
                valid_mask = valid_mask | ~np.isnan(scale_ds_ecoregion_clip[var])

            if np.count_nonzero(valid_mask) == 0:
                print(f'No valid data for region {NAm_ecoregion},  skipping...')
                continue

            # Get the x and y coordinates
            x_coords, y_coords = np.meshgrid(scale_ds_ecoregion_clip['x'].values, scale_ds_ecoregion_clip['y'].values, indexing='ij')
            x_flat = x_coords.flatten()[valid_mask.values.flatten()]
            y_flat = y_coords.flatten()[valid_mask.values.flatten()]

            # Prepare the dataframe
            df = pd.DataFrame({
                'x': x_flat,
                'y': y_flat,
            })
            
            for var in refl_vars:
                df[var] = scale_ds_ecoregion_clip[var].values.flatten()[valid_mask.values.flatten()]
            df['forest_fraction'] = forest_frac_ecoregion_clip.values.flatten()[valid_mask.values.flatten()]
            df['crop_fraction'] = crop_frac_ecoregion_clip.values.flatten()[valid_mask.values.flatten()]
            df['otherveg_fraction'] = otherveg_frac_ecoregion_clip.values.flatten()[valid_mask.values.flatten()]
            df['elevation'] = elev_ecoregion_clip.values.flatten()[valid_mask.values.flatten()]

            # Save the dataframe to CSV
            df.to_csv(csv_output_file, index=False)
            print(f'Saved scaling factors to: {csv_output_file}')

    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)  


if __name__ == '__main__':
   main()