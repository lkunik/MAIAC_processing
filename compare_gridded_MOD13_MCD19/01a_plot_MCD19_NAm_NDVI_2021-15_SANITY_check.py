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
import dask as da

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
import rasterio
import geopandas as gpd

#########################################
# Define Global Filepaths
#########################################
#%%


#########################################
# Define Global functions
#########################################

def td_min(td):
    return td.seconds//60

def td_sec(td):
    return td.seconds%60



####################################
### Set up map parameters
####################################



# Load the natural earth raster file
NE_raster_file = "/uufs/chpc.utah.edu/common/home/lin-group19/ltk/pyutil/NaturalEarth/NE2_50M_SR_W/NE2_50M_SR_W.tif"
NE_raster = rasterio.open(NE_raster_file)
vector_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/pyutil/NaturalEarth'
coastlines_file = os.path.join(vector_dir, 'ne_110m_coastline.shp')
lakes_file = os.path.join(vector_dir, 'ne_110m_lakes.shp')
rivers_file = os.path.join(vector_dir, 'ne_110m_rivers_lake_centerlines.shp')
countries_file = os.path.join(vector_dir, 'ne_110m_admin_0_countries.shp')
lambert_conformal_proj4 = '+proj=lcc +lon_0=-100 +lat_0=40 +lat_1=33 +lat_2=45 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'

lakes_gdf = gpd.read_file(lakes_file)
# gdf.to_crs(lambert_conformal_proj4).plot(ax=ax, facecolor='#2b8dd6', edgecolor='#14629c', linewidth=1, alpha=0.7)

MCD19_basedir = '/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/01d_rect/QCfilt_CloudFree_LowAOD_ClearAdj'
composite_string = '2021-15'

# MCD19_files = sorted(glob.glob(os.path.join(MCD19_basedir, '*/', f'MCD19*{composite_string}_rect.nc')))

### Load state boundaries
fn = shpreader.natural_earth(
   resolution='10m', category='cultural', 
   name='admin_1_states_provinces',
)

reader = shpreader.Reader(fn)
states = [x for x in reader.records() if x.attributes["admin"] == "United States of America"] # get all states in US
states_geom = cfeature.ShapelyFeature([x.geometry for x in states], ccrs.PlateCarree())



# Load Canadian provinces and Mexican states
fn_can = shpreader.natural_earth(
    resolution='10m', category='cultural', 
    name='admin_1_states_provinces',
)
reader_can = shpreader.Reader(fn_can)
canada = [x for x in reader_can.records() if x.attributes["admin"] == "Canada"]
mexico = [x for x in reader_can.records() if x.attributes["admin"] == "Mexico"]

canada_geom = cfeature.ShapelyFeature([x.geometry for x in canada], ccrs.PlateCarree())
mexico_geom = cfeature.ShapelyFeature([x.geometry for x in mexico], ccrs.PlateCarree())

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



# First, ensure coordinates are sorted and rounded to avoid floating point issues
def prep_dataset(ds):
    # Round coordinates to avoid floating point mismatches
    ds = ds.assign_coords(
        x=np.round(ds.x.values, 5),
        y=np.round(ds.y.values, 5)
    )
    # Sort by coordinates
    ds = ds.sortby(['x', 'y'])
    return ds

#########################################
# Define Global Variables and constants
#########################################

#########################################
# Begin main
#########################################
#%%
# def main():

# mark start time to keep track of elapsed
start_total = time.time()

MODIS_tiles = [	
    'h08v04',
    'h08v05',
    'h08v06',
    'h09v04',
    'h09v05',
    'h10v02',
    'h10v03',
    'h10v04',
    'h10v05',
    'h10v06',
    'h11v02',
    'h11v03',
    'h11v04',
    'h11v05',
    'h12v02',
    'h12v03',
    'h12v04',
    'h12v05',
    'h13v02',
    'h13v03',
    'h13v04',
    'h14v02',
    'h14v03'
    ]
QC_option = 'CloudFree_LowAOD_ClearAdj' # options are 'CloudAOD', 'Cloud', 'AOD', or 'None'


tile_dirs = [os.path.join(MCD19_basedir, tile) for tile in MODIS_tiles]

plot_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/plots/maps/North_America/'

#%%

# overall_CCI_min = 1
# overall_CCI_max = -1
# overall_NDVI_min = 1
# overall_NDVI_max = -1
# for itile, tile_dir in enumerate(tile_dirs):
#     MCD19A1_file = os.path.join(tile_dir, f'MCD19A1.{MODIS_tiles[itile]}.061_01d_annual_{avg_window}_CV-MVC_QCfilt_{QC_option}.nc')

#     if not os.path.exists(MCD19A1_file):
#         print(f'File not found: {MCD19A1_file}. Skipping this tile.')
#         continue
#     print(f'getting min and max from tile {MODIS_tiles[itile]}...')
#     dat_ds = xr.open_dataset(MCD19A1_file).drop_vars(['NDVI'])
#     dat_ds = dat_ds.assign(CCI=(dat_ds["Sur_refl11"] - dat_ds["Sur_refl1"])/
#             (dat_ds["Sur_refl11"] + dat_ds["Sur_refl1"]))
#     dat_ds = dat_ds.assign(NDVI=(dat_ds["Sur_refl2"] - dat_ds["Sur_refl1"])/
#             (dat_ds["Sur_refl2"] + dat_ds["Sur_refl1"]))

#     dat_ds_CCI_mean = dat_ds['CCI'].mean(dim='year', skipna=True)
#     dat_ds_NDVI_mean = dat_ds['NDVI'].mean(dim='year', skipna=True)

#     overall_CCI_min = float(np.min([np.nanquantile(dat_ds_CCI_mean.values, 0.01), overall_CCI_min]))
#     overall_CCI_max = float(np.max([np.nanquantile(dat_ds_CCI_mean.values, 0.99), overall_CCI_max]))
#     overall_NDVI_min = float(np.min([np.nanquantile(dat_ds_NDVI_mean.values, 0.01), overall_NDVI_min]))
#     overall_NDVI_max = float(np.max([np.nanquantile(dat_ds_NDVI_mean.values, 0.99), overall_NDVI_max]))

# print(f'Overall CCI min: {overall_CCI_min}, Overall CCI max: {overall_CCI_max}')
# print(f'Overall NDVI min: {overall_NDVI_min}, Overall NDVI max: {overall_NDVI_max}')
# Overall CCI min: -0.3658804343909615, Overall CCI max: 0.2306400408606525
# Overall NDVI min: 0.019227114090525473, Overall NDVI max: 0.9046409437773152

overall_NDVI_min = -0.4
overall_NDVI_max = 1


#%%

zlab = 'NDVI'
cmap_name = 'gnuplot_r'
# Set up Lambert figure
projection = ccrs.LambertConformal(central_longitude=-100, central_latitude=40)
fig, ax = plt.subplots(figsize=(30, 30), subplot_kw={'projection': projection})

# ax.set_extent(plot_extent, crs=ccrs.PlateCarree())  # PlateCarree is default lat/lon 
ax.set_extent([-170, -50, 9, 72], crs=ccrs.PlateCarree())

# ax.stock_img()
# After setting extent
img_extent = [NE_raster.bounds.left, NE_raster.bounds.right, 
              NE_raster.bounds.bottom, NE_raster.bounds.top]

ax.imshow(NE_raster.read([1,2,3]).transpose(1,2,0), 
          extent=img_extent,
          transform=ccrs.PlateCarree(),
          origin='upper', zorder=1, alpha=0.8)

# ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.7)
# ax.add_feature(cfeature.OCEAN, facecolor='black', edgecolor='black', linewidth=5, alpha=0.7, zorder = 3)
# Get ocean geometries and buffer them
ocean_feature = cfeature.NaturalEarthFeature('physical', 'ocean', '50m')
ocean_geoms = [geom.buffer(0.05) for geom in ocean_feature.geometries()]  # 0.05 degrees buffer
ocean_buffered = cfeature.ShapelyFeature(ocean_geoms, ccrs.PlateCarree())

ax.add_feature(ocean_buffered, facecolor='black', edgecolor='none', alpha=0.7, zorder=3)

# ax.add_feature(cfeature.COASTLINE, edgecolor='black', zorder = 5)
ax.add_feature(cfeature.LAKES, facecolor='#1C1E9E', edgecolor='#14629c', alpha = 0.7, zorder = 3)
# lakes_gdf.plot(ax=ax, facecolor='#1C1E9E', edgecolor='#14629c', linewidth=1, alpha=1, zorder = 5)

for itile, tile_dir in enumerate(tile_dirs):
    print(f'Plotting NDVI for tile {MODIS_tiles[itile]}...')

    #MCD19A1.h13v02.061_01d_16day_CV-MVC_QCfilt_CloudFree_LowAOD_ClearAdj_2021-15_rect.nc
    MCD19A1_file = os.path.join(tile_dir, f'MCD19A1.{MODIS_tiles[itile]}.061_01d_16day_CV-MVC_QCfilt_{QC_option}_{composite_string}_rect.nc')

    if not os.path.exists(MCD19A1_file):
        print(f'File not found: {MCD19A1_file}. Skipping this tile.')
        continue
    dat_ds = xr.open_dataset(MCD19A1_file)
    
    # dat_ds = dat_ds.assign(CCI=(dat_ds["Sur_refl11"] - dat_ds["Sur_refl1"])/
    #         (dat_ds["Sur_refl11"] + dat_ds["Sur_refl1"]))

    if MODIS_tiles[itile] in ['h15v01', 'h15v02', 'h16v01', 'h16v02']:
        dat_ds['NDVI'] = dat_ds['NDVI'].where((dat_ds['lon'] < -60) | (dat_ds['lat'] < 58), drop=False)
    
    im = dat_ds['NDVI'].plot(
        ax=ax, x='lon', y='lat', transform=ccrs.PlateCarree(),
        alpha=1, cmap=cmap_name,
        vmin=overall_NDVI_min, vmax=overall_NDVI_max,
        add_colorbar=False, zorder = 4
    )

ax.coastlines(resolution='50m', linewidth=1, zorder = 5)
# ax.plot(US_NR1_lonlat[0], US_NR1_lonlat[1], '*', markersize=30, markerfacecolor='yellow', markeredgecolor='black', markeredgewidth=1, transform=ccrs.PlateCarree())

# ax.add_feature(states_geom, facecolor="none", edgecolor="#737373", linewidth=0.4) # add state boundaries to the map
# ax.add_feature(canada_geom, facecolor="none", edgecolor="#737373", linewidth=0.4) # add Canadian province boundaries to the map
# ax.add_feature(mexico_geom, facecolor="none", edgecolor="#737373", linewidth=0.4) # add Mexican state boundaries to the map
ax.set_title('')
outfile = os.path.join(plot_dir, f'MCD19A1_NorthAmerica_NDVI_map_QCfilt_{QC_option}_lambert_01d_{composite_string}.png')
plt.savefig(outfile, format='png', dpi=500, bbox_inches='tight')
plt.close()


