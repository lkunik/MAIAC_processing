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

LC_basedir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/ESA_CCI/LandCover/for_MAIAC_proc/'
forest_frac_dir = os.path.join(LC_basedir, 'forest_fraction/')
crop_frac_dir = os.path.join(LC_basedir, 'crop_fraction/')
otherveg_frac_dir = os.path.join(LC_basedir, 'other_vegetation_fraction/')
elev_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/USGS/GTOPO30/for_MAIAC_proc/'
EPA_ecoregion_regrid_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/EPA/ecoregions/L3/for_MAIAC_proc/'

plot_basedir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/plots/QC/WUS_test/'
ecoregion_mask_plot_dir = os.path.join(plot_basedir, 'ecoregion_mask_byTile')
elev_plot_dir = os.path.join(plot_basedir, 'elev_byTile')
LC_plot_basedir = os.path.join(plot_basedir, 'LC_mask_byTile')
forest_plot_dir = os.path.join(LC_plot_basedir, 'forest_fraction')
crop_plot_dir = os.path.join(LC_plot_basedir, 'crop_fraction')
otherveg_plot_dir = os.path.join(LC_plot_basedir, 'other_vegetation_fraction')

Path(ecoregion_mask_plot_dir).mkdir(parents=True, exist_ok=True)
Path(elev_plot_dir).mkdir(parents=True, exist_ok=True)
Path(forest_plot_dir).mkdir(parents=True, exist_ok=True)
Path(crop_plot_dir).mkdir(parents=True, exist_ok=True)
Path(otherveg_plot_dir).mkdir(parents=True, exist_ok=True)

EPA_ecoregion_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/EPA/ecoregions/'
EPA_ecoregion_L3_file = os.path.join(EPA_ecoregion_dir, 'L3', 'WUS_cec_eco_l3.shp')

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
# plot_extent = [-128, -106, 39, 51]  # [minx, maxx, miny, maxy], bounds for western US


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

MODIS_tiles = [	
    'h08v04',
    'h08v05',
    'h09v04',
    'h09v05', 
    'h10v04',
    ]
QC_option = 'Cloud' # options are 'CloudAOD', 'Cloud', 'AOD', or 'None'
sample_year = 2018 # this is the specific year-period that every tile has a sample for
sample_period = 14


#########################################
# Begin main
#########################################
#%%

ecoregion_l3_gdf = gpd.read_file(EPA_ecoregion_L3_file)
WUS_ecoregions = np.unique(ecoregion_l3_gdf.NA_L3NAME)

# MODIS_tile = 'h11v02'
for MODIS_tile in MODIS_tiles:

    print(f'MODIS tile {MODIS_tile}')

    forest_frac_file = os.path.join(forest_frac_dir, f'forest_frac_1km_curv_{MODIS_tile}.nc')
    if not os.path.exists(forest_frac_file):
        print(f'No Forest fraction file for {MODIS_tile} , skipping...')
    else:

        print('')
        print('-~~~~~~~~~~~~~~~~~~-------~~~~~~~~~~~~~~~~~~-')
        print(f'plotting Forest land cover mask for tile {MODIS_tile}...')
        forest_frac_xr_curv = xr.open_dataset(forest_frac_file)['__xarray_dataarray_variable__']
        forest_frac_mapfile = os.path.join(forest_plot_dir, f'forest_frac_1km_curv_{MODIS_tile}.png')
        # plot_extent = get_tile_plot_extent(forest_frac_xr_curv)
        test_plot(forest_frac_xr_curv, plot_title='forest fraction', zlab='forest fraction', plot_filename=forest_frac_mapfile)
        print(f'Map of forest fraction for {MODIS_tile} saved to {forest_frac_mapfile}')
        print('-~~~~~~~~~~~~~~~~~~-------~~~~~~~~~~~~~~~~~~-')
        print('')


    crop_frac_file = os.path.join(crop_frac_dir, f'crop_frac_1km_curv_{MODIS_tile}.nc')
    if not os.path.exists(crop_frac_file):
        print(f'No Crop fraction file for {MODIS_tile} , skipping...')
    else:

        print('')
        print('-~~~~~~~~~~~~~~~~~~-------~~~~~~~~~~~~~~~~~~-')
        print(f'plotting Crop land cover mask for tile {MODIS_tile}...')
        crop_frac_xr_curv = xr.open_dataset(crop_frac_file)['__xarray_dataarray_variable__']
        crop_frac_mapfile = os.path.join(crop_plot_dir, f'crop_frac_1km_curv_{MODIS_tile}.png')
        test_plot(crop_frac_xr_curv, plot_title='crop fraction', zlab='crop fraction', plot_filename=crop_frac_mapfile)
        print(f'Map of crop fraction for {MODIS_tile} saved to {crop_frac_mapfile}')
        print('-~~~~~~~~~~~~~~~~~~-------~~~~~~~~~~~~~~~~~~-')
        print('')


    otherveg_frac_file = os.path.join(otherveg_frac_dir, f'otherveg_frac_1km_curv_{MODIS_tile}.nc')
    if not os.path.exists(otherveg_frac_file):
        print(f'No Other vegetation fraction file for {MODIS_tile}, skipping...')
    else:

        print('')
        print('-~~~~~~~~~~~~~~~~~~-------~~~~~~~~~~~~~~~~~~-')
        print(f'plotting Other Vegetation land cover mask for tile {MODIS_tile}...')
        otherveg_frac_xr_curv = xr.open_dataset(otherveg_frac_file)['__xarray_dataarray_variable__']
        otherveg_frac_mapfile = os.path.join(otherveg_plot_dir, f'otherveg_frac_1km_curv_{MODIS_tile}.png')
        test_plot(otherveg_frac_xr_curv, plot_title='other vegetation fraction', zlab='other vegetation fraction', plot_filename=otherveg_frac_mapfile)
        print(f'Map of other vegetation fraction for {MODIS_tile} saved to {otherveg_frac_mapfile}')
        print('-~~~~~~~~~~~~~~~~~~-------~~~~~~~~~~~~~~~~~~-')
        print('')





    elev_file = os.path.join(elev_dir, f'GTOPO30_DEM_1km_curv_{MODIS_tile}.nc')
    if not os.path.exists(elev_file):
        print(f'No Elevation file for {MODIS_tile}, skipping...')
    else:

        print('')
        print('-~~~~~~~~~~~~~~~~~~-------~~~~~~~~~~~~~~~~~~-')
        print(f'plotting elevation for tile {MODIS_tile}...')
        elev_xr_curv = xr.open_dataset(elev_file)['__xarray_dataarray_variable__']
        elev_mapfile = os.path.join(elev_plot_dir, f'GTOPO30_DEM_1km_curv_{MODIS_tile}.png')
        test_plot(elev_xr_curv, plot_title='elevation regrid', zlab='elevation', plot_filename=elev_mapfile)
        print(f'Map of elevation for {MODIS_tile} saved to {elev_mapfile}')
        print('-~~~~~~~~~~~~~~~~~~-------~~~~~~~~~~~~~~~~~~-')
        print('')


    # Loop through the ecoregion masks
    print('Looping through ecoregions to create masks...')

    ecoregion_mask_plot_dir_TILE = os.path.join(ecoregion_mask_plot_dir, f'{MODIS_tile}')
    Path(ecoregion_mask_plot_dir_TILE).mkdir(parents=True, exist_ok=True)
    WUS_ecoregion = WUS_ecoregions[0]
    for WUS_ecoregion in WUS_ecoregions:
        ecoregion_mask_file = os.path.join(EPA_ecoregion_regrid_dir, f'EPA_L3_ecoregion_mask_1km_curv_{MODIS_tile}_{WUS_ecoregion.replace(" ", "_").replace("/", "_")}.nc')

        if not os.path.exists(ecoregion_mask_file):
            print(f'No ecoregion mask for {WUS_ecoregion}, skipping...')
            continue
        
        ecoregion_mask_xr = xr.open_dataset(ecoregion_mask_file)['__xarray_dataarray_variable__']
        if (ecoregion_mask_xr == 0).all():
            print(f'Ecoregion mask is all zeros for {WUS_ecoregion}, skipping...')
            continue
        
        print('')
        print('-~~~~~~~~~~~~~~~~~~-------~~~~~~~~~~~~~~~~~~-')
        print(f'plotting ecoregion mask {WUS_ecoregion}...')
        ecoregion_mask_mapfile = os.path.join(ecoregion_mask_plot_dir_TILE, f'EPA_L3_ecoregion_mask_1km_curv_{MODIS_tile}_{WUS_ecoregion.replace(" ", "_").replace("/", "_")}.png')
        test_plot(ecoregion_mask_xr, plot_title=f'EPA L3 ecoregion mask: {WUS_ecoregion}', zlab='ecoregion mask', plot_filename=ecoregion_mask_mapfile)
        print(f'Map of ecoregion mask for {WUS_ecoregion} saved to {ecoregion_mask_mapfile}')
        print('-~~~~~~~~~~~~~~~~~~-------~~~~~~~~~~~~~~~~~~-')
        print('')
