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

US_NR1_lonlat = (-105.5464, 40.0329)

# # use Google Satellite imagery as basemap
# tiler = img_tiles.GoogleTiles(style='satellite')
# crs = tiler.crs # set crs of map tiler
# alpha = 0.5  # transparency 0-1
# transform = ccrs.PlateCarree()  # transform specifies the crs that the data is in
# plot_extent = [-128, -106, 39, 51]  # [minx, maxx, miny, maxy], bounds for western US

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

refl_vars = ['Sur_refl1', 'Sur_refl2', 'Sur_refl4', 'Sur_refl6', 'Sur_refl7', 'Sur_refl11']

#########################################
# Begin main
#########################################
#%%
# def main():

# mark start time to keep track of elapsed
start_total = time.time()

MODIS_tiles = [	
    'h07v06',
    'h08v03',  
    'h08v04',
    'h08v05',
	'h08v06',
	'h08v07', 
    'h09v02', 
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
    'h11v08',  
	'h12v02',
	'h12v03', 
	'h12v04',
    'h12v05',
	'h13v02', 
	'h13v03', 
	'h13v04',
	'h14v02',
    'h14v03',
    'h14v04',
    "h12v01",
    "h13v01",
    "h14v01",
    "h15v01",
    "h15v02",
    "h16v01"
    ]
QC_option = 'CloudFree_LowAOD_ClearAdj' # CHANGED: QC option used for the gap-fill QC files
plot_var = 'gapfill_CCI_frac' # CHANGED: variable to plot (mean annual number of composites gap-filled for CCI);
                               # other options: 'gapfill_NDVI_count', 'gapfill_CCI_frac', 'gapfill_NDVI_frac'

# CHANGED: annualized gap-fill QC files on the 1 km sinusoidal grid
dat_basedir = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/QC_gapfill/QCfilt_{QC_option}/'
tile_dirs = [os.path.join(dat_basedir, tile, 'annualized') for tile in MODIS_tiles]


plot_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/plots/maps/North_America/QC/'

#%%

# CHANGED: color limits computed from the data (0 to the largest per-tile 99th percentile), same idea as the
# commented-out min/max block in the CCI map script. Once printed, these can be hard-coded and this block commented out.
overall_min = 0
overall_max = 0

if overall_max == 0 and overall_min == 0:
    for itile, tile_dir in enumerate(tile_dirs):
        QC_file = os.path.join(tile_dir, f'MCD19A1.{MODIS_tiles[itile]}.061_1km_annual_CV-MVC_QCfilt_{QC_option}_gapfillQC.nc')

        if not os.path.exists(QC_file):
            continue
        with xr.open_dataset(QC_file) as dat_ds:
            overall_max = float(np.max([np.nanquantile(dat_ds[plot_var].values, 0.99), overall_max]))

    print(f'Overall {plot_var} min: {overall_min}, Overall {plot_var} max: {overall_max}')
else:
    print(f'Using pre-set min/max: Overall {plot_var} min: {overall_min}, Overall {plot_var} max: {overall_max}')

#%%

zlab = plot_var
cmap_name = 'inferno_r'#'magma_r' # CHANGED: color scheme
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

# CHANGED: ocean and lakes drawn above the data (zorder 4.5 instead of 3), because water pixels are 0 (not NaN)
# in the gap-fill QC files and would otherwise be colored as "no gap-filling"
ax.add_feature(ocean_buffered, facecolor='black', edgecolor='none', alpha=0.7, zorder=4.5)

# ax.add_feature(cfeature.COASTLINE, edgecolor='black', zorder = 5)
ax.add_feature(cfeature.LAKES, facecolor='#1C1E9E', edgecolor='#14629c', alpha = 0.7, zorder = 4.5)
# lakes_gdf.plot(ax=ax, facecolor='#1C1E9E', edgecolor='#14629c', linewidth=1, alpha=1, zorder = 5)

for itile, tile_dir in enumerate(tile_dirs):
    print(f'Plotting {plot_var} for tile {MODIS_tiles[itile]}...')
    QC_file = os.path.join(tile_dir, f'MCD19A1.{MODIS_tiles[itile]}.061_1km_annual_CV-MVC_QCfilt_{QC_option}_gapfillQC.nc')

    if not os.path.exists(QC_file):
        print(f'File not found: {QC_file}. Skipping this tile.')
        continue
    dat_ds = xr.open_dataset(QC_file)

    if MODIS_tiles[itile] in ['h15v01', 'h15v02', 'h16v01', 'h16v02']:
        dat_ds[plot_var] = dat_ds[plot_var].where((dat_ds['lon'] < -60) | (dat_ds['lat'] < 58), drop=False)

    # CHANGED: no year averaging needed, the file is already averaged across years
    dat_ds_mean = dat_ds.set_coords(['lon', 'lat'])
    

    # dat_ds_CCI_mean.plot(ax=ax, x='x', y='y', transform=ccrs.PlateCarree(), 
    #                 alpha=1, cmap='YlGn',
    #                 vmin=overall_CCI_min, vmax=overall_CCI_max,
    #                 add_colorbar=False)
    im = dat_ds_mean[plot_var].plot(
        ax=ax, x='lon', y='lat', transform=ccrs.PlateCarree(),
        alpha=1, cmap=cmap_name,
        vmin=overall_min, vmax=overall_max,
        add_colorbar=False, zorder = 4
    )

ax.coastlines(resolution='50m', linewidth=1, zorder = 5)

# colorbar inset in the lower-left corner of the map, over the dark ocean backdrop (hence white text)
cax = ax.inset_axes([0.04, 0.14, 0.3, 0.03], zorder=10)  # [x0, y0, width, height] in axes-fraction coordinates
cbar = fig.colorbar(plt.cm.ScalarMappable(norm=mcolors.Normalize(vmin=overall_min, vmax=overall_max), cmap=cmap_name),
                    cax=cax, orientation='horizontal', extend='max')  # 'max' because overall_max is a 99th-percentile cap
cbar.set_label(zlab, color='white', fontsize=36, labelpad=12)
cbar.ax.tick_params(colors='white', labelsize=30, length=8, width=2)
cbar.outline.set_edgecolor('white')
cbar.outline.set_linewidth(2)
# ax.plot(US_NR1_lonlat[0], US_NR1_lonlat[1], '*', markersize=30, markerfacecolor='yellow', markeredgecolor='black', markeredgewidth=1, transform=ccrs.PlateCarree())

# ax.add_feature(states_geom, facecolor="none", edgecolor="#737373", linewidth=0.4) # add state boundaries to the map
# ax.add_feature(canada_geom, facecolor="none", edgecolor="#737373", linewidth=0.4) # add Canadian province boundaries to the map
# ax.add_feature(mexico_geom, facecolor="none", edgecolor="#737373", linewidth=0.4) # add Mexican state boundaries to the map
ax.set_title('')
outfile = os.path.join(plot_dir, f'MCD19A1_NorthAmerica_annual_{plot_var}_map_QCfilt_{QC_option}_lambert_1km_{cmap_name}.png')
plt.savefig(outfile, format='png', dpi=500, bbox_inches='tight')
plt.close()

print(f'{plot_var} map complete.')