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

LC_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/ESA_CCI/LandCover/'
forest_frac_file = os.path.join(LC_dir, 'forest_fraction_0.01deg_north_america.nc')
crop_frac_file = os.path.join(LC_dir, 'crop_fraction_0.01deg_north_america.nc')
otherveg_frac_file = os.path.join(LC_dir, 'other_vegetation_fraction_0.01deg_north_america.nc')

elev_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/USGS/GTOPO30/01d/'
elev_file = os.path.join(elev_dir, 'GTOPO30_DEM_01d_NAm.nc')

LC_out_basedir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/ESA_CCI/LandCover/for_MAIAC_proc/'

forest_frac_outdir = os.path.join(LC_out_basedir, 'forest_fraction/')
crop_frac_outdir = os.path.join(LC_out_basedir, 'crop_fraction/')
otherveg_frac_outdir = os.path.join(LC_out_basedir, 'other_vegetation_fraction/')
Path(forest_frac_outdir).mkdir(parents=True, exist_ok=True)
Path(crop_frac_outdir).mkdir(parents=True, exist_ok=True)
Path(otherveg_frac_outdir).mkdir(parents=True, exist_ok=True)

elev_out_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/USGS/GTOPO30/for_MAIAC_proc/'
Path(elev_out_dir).mkdir(parents=True, exist_ok=True)


# EPA_ecoregion_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/EPA/ecoregions/L3/NAm_shp_FOR_MAIAC_PROC/'
# EPA_ecoregion_L3_file = os.path.join(EPA_ecoregion_dir, 'NAm_cec_eco_l3_v2.shp')
# EPA_ecoregion_regrid_out_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/EPA/ecoregions/L3/for_MAIAC_proc/NAm/'
EPA_ecoregion_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/EPA/ecoregions/L2/NAm_shp_FOR_MAIAC_PROC/'
EPA_ecoregion_L2_file = os.path.join(EPA_ecoregion_dir, 'NA_CEC_Eco_Level2_withOther.shp')
EPA_ecoregion_regrid_out_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/EPA/ecoregions/L2/for_MAIAC_proc/NAm/'

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



#########################################
# Define Global Variables and constants
#########################################

MODIS_tiles = [	
    'h07v06',  
    'h08v04',
    'h08v05',
	'h08v06',
	'h08v07', 
    'h09v02', #<----------------THIS ONE HAS ISSUES REGRIDDING... never completes. Likely because crosses 180 meridian
	'h09v03',  
    'h09v04',
    'h09v05', 
	'h09v06',
	'h09v07', 
	'h09v08',  
	'h10v02', 
	'h10v03',
    'h10v04',
	'h10v05',
	'h10v06', 
	'h10v07',
	'h10v08', 
	'h11v02', 
	'h11v03', 
	'h11v04', 
	'h11v05', 
    'h11v06', 
    'h11v07',  
	'h12v02',
	'h12v03', 
	'h12v04',
	'h13v02', 
	'h13v03', 
	'h13v04',
	'h14v02',
    'h14v03'
    ]
QC_option = 'Cloud' # options are 'CloudAOD', 'Cloud', 'AOD', or 'None'
sample_year = 2018 # this is the specific year-period that every tile has a sample for
sample_period = 14

hist_years = np.arange(2000, 2025)
DOY_composite_starts = np.arange(1, 366, 16)
hist_composites = range(len(DOY_composite_starts))  # 0-22

refl_vars = ['Sur_refl1', 'Sur_refl2', 'Sur_refl4', 'Sur_refl6', 'Sur_refl7', 'Sur_refl11']
#########################################
# Begin main
#########################################
#%%
# def main():

# ecoregion_l3_gdf = gpd.read_file(EPA_ecoregion_L3_file)
# NAm_ecoregions = np.unique(ecoregion_l3_gdf.NA_L3NAME)
ecoregion_l2_gdf = gpd.read_file(EPA_ecoregion_L2_file)
NAm_ecoregions = np.unique(ecoregion_l2_gdf.NA_L2NAME)

for MODIS_tile in MODIS_tiles:
    print(f'Tile: {MODIS_tile}')
    for NAm_ecoregion in NAm_ecoregions:
        ecoregion_mask_outfile = os.path.join(EPA_ecoregion_regrid_out_dir, f'EPA_L2_ecoregion_mask_1km_curv_{MODIS_tile}_{NAm_ecoregion.replace(" ", "_").replace("/", "_")}.nc')
        
        if os.path.exists(ecoregion_mask_outfile):
            print(f'Ecoregion mask file found: {NAm_ecoregion}')

