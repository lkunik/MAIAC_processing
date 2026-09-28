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

NLCD_TCC_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/NLCD/TreeCover/2011/'
NLCD_file = os.path.join(NLCD_TCC_dir, 'NLCD_TCC_1km_WUS_2011.nc')

SRTM_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/SRTM/WUS'
SRTM_file = os.path.join(SRTM_dir, 'SRTM_DEM_0083d_WUS.nc')

TCC_out_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/NLCD/TreeCover/2011/for_MAIAC_proc/'
Path(TCC_out_dir).mkdir(parents=True, exist_ok=True)

SRTM_out_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/SRTM/WUS/for_MAIAC_proc/'
Path(SRTM_out_dir).mkdir(parents=True, exist_ok=True)


EPA_ecoregion_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/EPA/ecoregions/'
EPA_ecoregion_L3_file = os.path.join(EPA_ecoregion_dir, 'L3', 'WUS_cec_eco_l3.shp')
EPA_ecoregion_regrid_out_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/EPA/ecoregions/L3/for_MAIAC_proc/'

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
plot_extent = [-128, -106, 39, 51]  # [minx, maxx, miny, maxy], bounds for western US

def test_plot(dat_da, plot_title, zlab, plot_filename = None, plot_min = None, plot_max = None):
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


hist_years = np.arange(2000, 2025)
DOY_composite_starts = np.arange(1, 366, 16)
hist_composites = range(len(DOY_composite_starts))  # 0-22

refl_vars = ['Sur_refl1', 'Sur_refl2', 'Sur_refl4', 'Sur_refl6', 'Sur_refl7', 'Sur_refl11']
#########################################
# Begin main
#########################################
#%%
# def main():




# Load in TCC dataset and regrid to 1km curvilinear grid
TCC_xr_raw = xr.open_dataset(NLCD_file).squeeze('band')['tree_canopy_cover']

elev_xr_raw = xr.open_dataset(SRTM_file).squeeze('band')['elevation']


ecoregion_l3_gdf = gpd.read_file(EPA_ecoregion_L3_file)

WUS_ecoregions = np.unique(ecoregion_l3_gdf.NA_L3NAME)
# ['Arizona/New Mexico Mountains', 'Arizona/New Mexico Plateau',
# 'Aspen Parkland/Northern Glaciated Plains', 'Blue Mountains',
# 'California Coastal Sage, Chaparral, and Oak Woodlands',
# 'Canadian Rockies', 'Cascades', 'Central Basin and Range',
# 'Central California Valley', 'Chihuahuan Desert',
# 'Chihuahuan Deserts', 'Coast Range', 'Colorado Plateaus',
# 'Columbia Mountains/Northern Rockies', 'Columbia Plateau',
# 'Eastern Cascades Slopes and Foothills', 'High Plains',
# 'Idaho Batholith', 'Klamath Mountains', 'Madrean Archipelago',
# 'Middle Rockies', 'Mojave Basin and Range', 'Nebraska Sand Hills',
# 'North Cascades', 'Northern Basin and Range',
# 'Northwestern Glaciated Plains', 'Northwestern Great Plains',
# 'Sierra Nevada', 'Snake River Plain', 'Sonoran Desert',
# 'Southern Rockies',
# 'Southern and Baja California Pine-Oak Mountains',
# 'Southwestern Tablelands', 'Strait of Georgia/Puget Lowland',
# 'Wasatch and Uinta Mountains', 'Willamette Valley',
# 'Wyoming Basin']

# ecoregion_l3_gdf_ENF = ecoregion_l3_gdf[ecoregion_l3_gdf.NA_L3NAME.isin(WUS_ecoregions)]
# ecoregion_l3_geom_ENF = ecoregion_l3_gdf_ENF.geometry

# set up blank ecoregion masks on the 1km curvilinear grid
xr_ecoregion_mask = xr.zeros_like(elev_xr_raw, dtype=int)
xr_ecoregion_mask = xr_ecoregion_mask.rio.write_crs("EPSG:4326")


MODIS_tiles = ['h09v04', 'h10v04', 'h08v04','h09v05', 'h08v05']
QC_option = 'CloudFree' # options are 'CloudAOD', 'Cloud', 'AOD', or 'None'

sample_year = 2015
sample_period = 12

MODIS_tile = 'h08v04'
for MODIS_tile in MODIS_tiles:

    print(f'forming regridder for MODIS tile {MODIS_tile}')
    dat_dir = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/pre-fill/QCfilt_{QC_option}/{MODIS_tile}/'
    MCD19A1_files = sorted(glob.glob(os.path.join(dat_dir, f'MCD19A1.{MODIS_tile}.*.nc')))

    sample_file = os.path.join(dat_dir, f'MCD19A1.{MODIS_tile}.061_01d_16day_CV-MVC_QCfilt_{QC_option}_{sample_year}-{sample_period}.nc')
    sample_1km_curv_ds = xr.open_dataset(sample_file).squeeze('time')['NDVI']

    if MODIS_tile == 'h08v04':
        sample_1km_curv_ds = sample_1km_curv_ds.where(~np.isnan(sample_1km_curv_ds), drop=True)

    TCC_out_file = os.path.join(TCC_out_dir, f'TCC_1km_curv_{MODIS_tile}.nc')
    if os.path.exists(TCC_out_file):
        print(f'TCC regridder for {MODIS_tile} already exists, skipping...')
    else:
        print('TCC regridder...')
        regridder_TCC = xe.Regridder(TCC_xr_raw, sample_1km_curv_ds, "nearest_s2d", unmapped_to_nan = True) # Use XESMF package to establish a regridder which we can recycle
        TCC_xr_curv = regridder_TCC(TCC_xr_raw)
        TCC_xr_curv.to_netcdf(TCC_out_file)
        print('TCC regridder complete.')

        del regridder_TCC, TCC_xr_curv
        gc.collect()

    # Load in Elevation dataset and regrid to 1km curvilinear grid
    elev_out_file = os.path.join(SRTM_out_dir, f'SRTM_DEM_1km_curv_{MODIS_tile}.nc')
    if os.path.exists(elev_out_file):
        print(f'Elevation regridder for {MODIS_tile} already exists, skipping...')
    else:
        print('Elevation regridder...')
        regridder_elev = xe.Regridder(elev_xr_raw, sample_1km_curv_ds, "nearest_s2d", unmapped_to_nan = True)
        elev_xr_curv = regridder_elev(elev_xr_raw)
        elev_xr_curv.to_netcdf(elev_out_file)
        print('Elevation regridder complete.')

        # test_plot(elev_xr_curv, plot_title='elevation regrid', zlab='elevation')
        del elev_xr_curv
        gc.collect()

    # Loop through the ecoregions and create masks
    print('Looping through ecoregions to create masks...')
    WUS_ecoregion='Klamath Mountains'
    for WUS_ecoregion in WUS_ecoregions:
        ecoregion_mask_outfile = os.path.join(EPA_ecoregion_regrid_out_dir, f'EPA_L3_ecoregion_mask_1km_curv_{MODIS_tile}_{WUS_ecoregion.replace(" ", "_").replace("/", "_")}.nc')

        if os.path.exists(ecoregion_mask_outfile):
            print(f'Ecoregion mask for {WUS_ecoregion} already exists, skipping...')
            continue

        print(f'Ecoregion masking for {WUS_ecoregion}')
        region_geom = ecoregion_l3_gdf[ecoregion_l3_gdf.NA_L3NAME == WUS_ecoregion].geometry
        
        xr_ecoregion_mask_clip = xr_ecoregion_mask.rio.clip(region_geom, drop=False, invert=False)

        # Check if the mask is empty after clipping
        if xr_ecoregion_mask_clip.count().item() == 0:
            print(f'Ecoregion mask for {WUS_ecoregion} is empty after clipping, skipping...')
            continue

        # Set all invalid values in xr_ecoregion_mask_clip to -1
        
        xr_ecoregion_mask_clip = xr_ecoregion_mask_clip.where(xr_ecoregion_mask_clip > -1, -1)
        xr_ecoregion_mask_clip+=1  # set invalid values to 0, valid values to 1

        xr_ecoregion_mask_curv = regridder_elev(xr_ecoregion_mask_clip)
        xr_ecoregion_mask_curv = xr_ecoregion_mask_curv.where(xr_ecoregion_mask_curv >=0, np.nan)
        
        # test_plot(xr_ecoregion_mask_curv, plot_title='ecoregion_mask', zlab='mask')
        # save to file
        xr_ecoregion_mask_curv.to_netcdf(ecoregion_mask_outfile)
        print(f'Ecoregion mask for {WUS_ecoregion} saved to {ecoregion_mask_outfile}')

    del regridder_elev
    gc.collect()
