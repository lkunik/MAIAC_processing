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

MOD13_basedir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/MODIS/NDVI_EVI_MOD13A2/01d_rect'
plot_year = 2021
plot_period = 15
MOD13_files = glob.glob(f'{MOD13_basedir}/*_{str(plot_year)}.nc')



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


plot_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/plots/maps/North_America/'

#%%

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

for MOD13_file in MOD13_files:
    print(f'Plotting NDVI for file {MOD13_file}...')

    dat_ds = xr.open_dataset(MOD13_file).isel(time = plot_period)
    

    # dat_ds_CCI_mean.plot(ax=ax, x='x', y='y', transform=ccrs.PlateCarree(), 
    #                 alpha=1, cmap='YlGn',
    #                 vmin=overall_CCI_min, vmax=overall_CCI_max,
    #                 add_colorbar=False)
    im = dat_ds['NDVI'].plot(
        ax=ax, x='x', y='y', transform=ccrs.PlateCarree(),
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
outfile = os.path.join(plot_dir, f'MOD13_NorthAmerica_NDVI_map_lambert_01d_{str(plot_year)}-{str(plot_period)}.png')
plt.savefig(outfile, format='png', dpi=500, bbox_inches='tight')
plt.close()


