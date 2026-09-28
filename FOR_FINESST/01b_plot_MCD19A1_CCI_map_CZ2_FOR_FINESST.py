#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Lewis Kunik - University of Utah
# Contact: lewis.kunik@utah.edu
#

#########################################
# Load packages
#########################################
#%%
# File system packages
import os  # operating system library
import glob
from pathlib import Path
import copy
import sys 
# runtime packages
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Custom packages
utils_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/pyutil'
if utils_dir not in sys.path:
   sys.path.append(utils_dir)
import data_utils
import plot_utils

# numerical/data management packages
import math
import numpy as np
import xarray as xr  # for multi dimensional data
import rioxarray as rxr
import pandas as pd
import xesmf as xe

# shapefile/geospatial packages
import geopandas as gpd
from shapely.geometry import mapping

# time and date packages
import time
from datetime import datetime as dt  # date time library
from datetime import timedelta

# plotting packages
import matplotlib
import pyreadr
from matplotlib import pyplot as plt  # primary plotting module
import matplotlib.colors as mcolors
import cartopy.crs as ccrs  # spatial plotting library 
from cartopy.io import img_tiles  # cartopy's implementation of webtiles
from matplotlib.ticker import FuncFormatter as FFmt  # need this for formatting plot ticks
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.path import Path as mplPath
from matplotlib.collections import PatchCollection

#%%
#########################################
# Define Global Filepaths
#########################################

# directories
dat_basedir = '/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/withCloudAODQC/by_tile/h08v05/'
polygon_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/disturbance/data/flux_tower_polygons/'
plot_basedir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/disturbance/plots/'
NLCD_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/NLCD/LandCover/regrid'
SRTM_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/SRTM/WUS'
# specify string to match for files using glob.glob (unix style pattern matching)
# If dataset is contained in a single file, just put the file basename
file_pattern = '*.nc'
nlcd_file = os.path.join(NLCD_dir, 'NLCD2021_evergreen_wus_0083d.nc')
srtm_file = os.path.join(SRTM_dir, 'SRTM_DEM_01d_WUS.nc')
########################################
# Define Global Variables and constants
#########################################
#%%

# Define domain bounding box
lon_bb_min = -119.4
lon_bb_max = -118.7
lat_bb_min = 36.8
lat_bb_max = 37.2

### Variables specific to the dataset 
plot_var = 'CCI' # this is the name of the variable to be accessed in the dataset
keep_vars = ['CCI', 'NDVI']

US_CZ2_coords = [-119.2566, 37.0311]
predrought_year = 2011
drought_year = 2014

elev_cutoff_low = 850
elev_cutoff_high = 3000

#%%
#########################################
# Define Global Functions
#########################################


# matplotlib map plotting functions
def z_axis_formatter(x, pos, deci=2):
   '''Format ticks to format with deci number of decimals'''

   return f'{x:.{deci}f}'

def add_lat_ticks(ax, ylims, labelsize=None, more_ticks=0):
   import cartopy.crs as ccrs
   from cartopy.mpl.ticker import LatitudeFormatter, LatitudeLocator

   fig = ax.figure
   bins = (fig.get_size_inches()[1] * fig.dpi / 100).astype(int) + 1

   y_ticks = LatitudeLocator(nbins=bins + more_ticks, prune='both')\
      .tick_values(ylims[0], ylims[1])

   ax.set_yticks(y_ticks, crs=ccrs.PlateCarree())
   ax.yaxis.tick_left()
   ax.yaxis.set_major_formatter(LatitudeFormatter())

   if labelsize is not None:
      ax.tick_params(axis='y', labelsize=labelsize)


def add_lon_ticks(ax, xlims, rotation=0, labelsize=None, more_ticks=0):
   import cartopy.crs as ccrs
   from cartopy.mpl.ticker import LongitudeFormatter, LongitudeLocator

   fig = ax.figure
   bins = (fig.get_size_inches()[0] * fig.dpi / 100).astype(int) + 1

   x_ticks = LongitudeLocator(nbins=bins + more_ticks, prune='both')\
      .tick_values(xlims[0], xlims[1])

   ax.set_xticks(x_ticks, crs=ccrs.PlateCarree())
   ax.xaxis.tick_bottom()
   ax.xaxis.set_major_formatter(LongitudeFormatter())

   if rotation != 0:
      ax.tick_params(axis='x', labelrotation=rotation)

   if labelsize is not None:
      ax.tick_params(axis='x', labelsize=labelsize)


def add_latlon_ticks(ax, extent, x_rotation=0, labelsize=None,
                     more_lon_ticks=0, more_lat_ticks=0):
   xlims, ylims = extent[:2], extent[2:]

   add_lat_ticks(ax, ylims, labelsize=labelsize, more_ticks=more_lat_ticks)

   add_lon_ticks(ax, xlims, rotation=x_rotation, labelsize=labelsize,
                  more_ticks=more_lon_ticks)


#########################################
# Begin Main
#########################################

#########################################
### Begin with standard argument handling
#########################################


#%%
#########################################
# Initialize map data
#########################################
# use Google Satellite imagery as basemap
tiler = img_tiles.GoogleTiles(style='satellite')
crs = tiler.crs # set crs of map tiler
alpha = 0.8  # transparency 0-1
transform = ccrs.PlateCarree()  # transform specifies the crs that the data is in

tiler_zoom = 12  # define a zoom level of detail
extent = [lon_bb_min, lon_bb_max, lat_bb_min, lat_bb_max] # [minx, maxx, miny, maxy]




#%%
#########################################
# Load Land Cover Data
#########################################

ENF_xr = xr.open_dataset(nlcd_file).sortby(['y', 'x'])
ENF_xr_clip = ENF_xr.sel(x = slice(lon_bb_min, lon_bb_max), y = slice(lat_bb_min, lat_bb_max))

DEM_xr = xr.open_dataset(srtm_file)
DEM_xr_clip = DEM_xr.sel(lon = slice(lon_bb_min, lon_bb_max), lat = slice(lat_bb_min, lat_bb_max))
#%%

# use pre-defined file search function to find relevant files
dat_files = glob.glob(f'{dat_basedir}*.nc')
# Filter dat_files for those matching any of the values in the specified array
years_to_include = list(range(2008, 2017))

dat_files_jul = [f for f in dat_files if any(f'{year}-07' in f for year in years_to_include)]
dat_files_aug = [f for f in dat_files if any(f'{year}-08' in f for year in years_to_include)]
dat_files_sept = [f for f in dat_files if any(f'{year}-09' in f for year in years_to_include)]
dat_files_combined = sorted(dat_files_jul + dat_files_aug + dat_files_sept)
# dat_files_combined = sorted(dat_files_aug + dat_files_sept)
############################
### Load satellite data
############################

#%%
dat_xr_arr = [] # set empty array

for dat_file in dat_files_combined:
   
   dat_xr = rxr.open_rasterio(dat_file).sortby(['y', 'x'])
   dat_xr = dat_xr[keep_vars]
   dat_xr.rio.write_crs(4326, inplace=True)
   # dat_xr = dat_xr.rename({'lon':'x', 'lat':'y'})

   # # convert the time coordinates to datetime date (see here: https://stackoverflow.com/questions/55786995/converting-cftime-datetimejulian-to-datetime)
   datetimeindex = dat_xr.indexes['time'].to_datetimeindex()
   dat_xr['time'] = datetimeindex

   dat_xr_clip = dat_xr.sel(x = slice(lon_bb_min, lon_bb_max), y = slice(lat_bb_min, lat_bb_max))
   
   num_clipcells = np.count_nonzero(~np.isnan(dat_xr_clip[plot_var].mean(dim = 'time')))

   print(f'number of grid cells used in clipping = {num_clipcells}')
   dat_xr_arr.append(dat_xr_clip)

# after looping through all files, combine all at the time dimension
dat_ds = xr.concat(dat_xr_arr, dim = 'time') # concatenate along time axis

# dat_ds = dat_ds[keep_vars]



#%%

if not data_utils.check_if_xr_grids_equal(ENF_xr_clip, dat_ds):
   ENF_xr_clip_ll = ENF_xr_clip.rename({'x':'lon', 'y':'lat'})
   dat_ds_ll = dat_ds.rename({'x':'lon', 'y':'lat'})
   regridder = xe.Regridder(ENF_xr_clip_ll, dat_ds_ll, 'conservative') # Set up regridder to match TROPOMI resolution
   ENF_xr_regrid = regridder(ENF_xr_clip_ll, keep_attrs=True, skipna = True)
else:
   ENF_xr_regrid = ENF_xr_clip

ENF_xr_regrid = ENF_xr_regrid.rename({'lon':'x', 'lat':'y'})


#%%

if not data_utils.check_if_xr_grids_equal(DEM_xr_clip, dat_ds):
   dat_ds_ll = dat_ds.rename({'x':'lon', 'y':'lat'})
   regridder = xe.Regridder(DEM_xr_clip, dat_ds_ll, 'conservative') # Set up regridder to match TROPOMI resolution
   DEM_xr_regrid = regridder(DEM_xr_clip, keep_attrs=True, skipna = True)
else:
   DEM_xr_regrid = DEM_xr_clip

DEM_xr_regrid = DEM_xr_regrid.rename({'lon':'x', 'lat':'y'})


#%%
# Plot ENF fraction map
fig, ax = plt.subplots(subplot_kw={'projection': crs})  # establish the figure, axes

# Set extent of map before adding base img
ax.set_extent(extent, crs=ccrs.PlateCarree())  # PlateCarree is default lat/lon

# Add base image
ax.add_image(tiler, tiler_zoom, interpolation='none')

ENF_xr_regrid['fraction_ENF'].plot(ax=ax, transform=transform, alpha=alpha,
                           cmap='Greens', vmin=0, vmax=1,
                           cbar_kwargs={'orientation': 'vertical',
                                     'pad': 0.1,
                                     'label': 'Fraction ENF',
                                     'shrink': 0.75,
                                     'format': FFmt(z_axis_formatter)})

# Add the US-CZ2 Flux Tower point
ax.plot(US_CZ2_coords[0], US_CZ2_coords[1], marker='*', color='black', markersize=10, transform=transform)
ax.text(US_CZ2_coords[0] + 0.02, US_CZ2_coords[1], 'US-CZ2', transform=transform, fontsize=12, color='black')

add_latlon_ticks(ax, extent, x_rotation=30, labelsize=8)
plt.xlabel(None)
plt.ylabel(None)
plt.title('ENF Fraction Map')
plt.show()



#%%
# Plot DEM
fig, ax = plt.subplots(subplot_kw={'projection': crs})  # establish the figure, axes

# Set extent of map before adding base img
ax.set_extent(extent, crs=ccrs.PlateCarree())  # PlateCarree is default lat/lon

# Add base image
ax.add_image(tiler, tiler_zoom, interpolation='none')

DEM_xr_regrid['elevation'].plot(ax=ax, transform=transform, alpha=alpha,
                           cmap='terrain',
                           cbar_kwargs={'orientation': 'vertical',
                                     'pad': 0.1,
                                     'label': 'elevation (m)',
                                     'shrink': 0.75,
                                     'format': FFmt(z_axis_formatter)})

# Add the US-CZ2 Flux Tower point
ax.plot(US_CZ2_coords[0], US_CZ2_coords[1], marker='*', color='black', markersize=10, transform=transform)
ax.text(US_CZ2_coords[0] + 0.02, US_CZ2_coords[1], 'US-CZ2', transform=transform, fontsize=12, color='black')

add_latlon_ticks(ax, extent, x_rotation=30, labelsize=8)
plt.xlabel(None)
plt.ylabel(None)
plt.title('SRTM DEM')
plt.show()

#%%
# Ensure the time index is sorted
dat_ds = dat_ds.sortby('time')

# Aggregate dat_ds by year, taking the mean
dat_ds_yearly = dat_ds.resample(time='1Y').mean(skipna=True)

# Apply an elevation filter
dat_ds_yearly_midElev = dat_ds_yearly.where(DEM_xr_regrid['elevation'] > elev_cutoff_low)
dat_ds_yearly_midElev = dat_ds_yearly_midElev.where(DEM_xr_regrid['elevation'] < elev_cutoff_high)

dat_ds_predrought = dat_ds_yearly_midElev.sel(time=str(predrought_year))
dat_ds_drought = dat_ds_yearly_midElev.sel(time=str(drought_year))

zmin = np.nanmin([dat_ds_predrought[plot_var].values, dat_ds_drought[plot_var].values])
zmax = np.nanmax([dat_ds_predrought[plot_var].values, dat_ds_drought[plot_var].values])
# zmin = np.nanmin(dat_ds_yearly_midElev[plot_var].values)
# zmax = np.nanmax(dat_ds_yearly_midElev[plot_var].values)

# #%%
# for yyyy in range(2008, 2017):
#    dat_ds_yyyy = dat_ds_yearly_midElev.sel(time=str(yyyy))[plot_var]

#    fig, ax = plt.subplots(subplot_kw={'projection': crs}) # establish the figure, axes

#    # Set extent of map before adding base img
#    ax.set_extent(extent, crs=ccrs.PlateCarree())  # PlateCarree is default lat/lon

#    # Add base image
#    ax.add_image(tiler, tiler_zoom, interpolation='none')

#    dat_ds_yyyy.plot(ax=ax, transform=transform, alpha=alpha,
#                      cmap='viridis_r', vmin = zmin, vmax = zmax,
#                      cbar_kwargs={'orientation': 'vertical',
#                                              'pad': 0.1,
#                                              'label': 'CCI',
#                                              'shrink': 0.75,
#                                              'format': FFmt(z_axis_formatter)})

#    # Add the US-CZ2 Flux Tower point
#    ax.plot(US_CZ2_coords[0], US_CZ2_coords[1], marker='*', color='black', markersize=10, transform=transform)
#    ax.text(US_CZ2_coords[0] + 0.02, US_CZ2_coords[1], 'US-CZ2', transform=transform, fontsize=12, color='black')
#    # plot the outline of the polygon
#    # ax.add_geometries(polygon_gdf.geometry, crs=ccrs.PlateCarree(), edgecolor=line_col, facecolor='none', linewidth=1.5)

#    add_latlon_ticks(ax, extent, x_rotation=30, labelsize = 8)
#    plt.xlabel(None)
#    plt.ylabel(None)
#    plt.title(f'July-Sept mean {yyyy}')
#    plt.show()



#%%
### PLOT PRE-DROUGHT
print(f'plotting overall annual mean map')
fig, ax = plt.subplots(subplot_kw={'projection': crs}) # establish the figure, axes

# Set extent of map before adding base img
ax.set_extent(extent, crs=ccrs.PlateCarree())  # PlateCarree is default lat/lon

# Add base image
ax.add_image(tiler, tiler_zoom, interpolation='none')

dat_ds_predrought[plot_var].plot(ax=ax, transform=transform, alpha=alpha,
                  cmap='viridis_r', vmin = zmin, vmax = zmax,
                  add_colorbar=False)

dat_ds_forcontour = dat_ds_predrought[plot_var].squeeze(drop=True)
# Draw a single contour around the non-nan values
contour = ax.contour(dat_ds_forcontour['x'], dat_ds_forcontour['y'], 
              np.isnan(dat_ds_forcontour), 
              levels=[0.5], colors='gray', transform=transform)

# Add the US-CZ2 Flux Tower point
ax.plot(US_CZ2_coords[0], US_CZ2_coords[1], marker='*', color='black', markersize=15, transform=transform)
ax.text(US_CZ2_coords[0] + 0.02, US_CZ2_coords[1], 'Flux Tower', transform=transform, fontsize=18, color='black')
# plot the outline of the polygon
# ax.add_geometries(polygon_gdf.geometry, crs=ccrs.PlateCarree(), edgecolor=line_col, facecolor='none', linewidth=1.5)

# add_latlon_ticks(ax, extent, x_rotation=30, labelsize = 8)
plt.xlabel(None)
plt.ylabel(None)
# plt.title(f'Summer {predrought_year}', fontsize = 16)
plt.title('')
plt.show()




#%%
##### PLOT DROUGHT YEAR

print(f'plotting overall annual mean map')
# fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': crs}) # establish the figure, axes with larger size
fig, ax = plt.subplots(subplot_kw={'projection': crs}) # establish the figure, axes

# Set extent of map before adding base img
ax.set_extent(extent, crs=ccrs.PlateCarree())  # PlateCarree is default lat/lon

# Add base image
ax.add_image(tiler, tiler_zoom, interpolation='none')

dat_ds_drought[plot_var].plot(ax=ax, transform=transform, alpha=alpha,
                  cmap='viridis_r', vmin = zmin, vmax = zmax,
                  add_colorbar=False)

dat_ds_forcontour = dat_ds_drought[plot_var].squeeze(drop=True)
# Draw a single contour around the non-nan values
contour = ax.contour(dat_ds_forcontour['x'], dat_ds_forcontour['y'], 
              np.isnan(dat_ds_forcontour), #linewidths=4,
              levels=[0.5], colors='gray', transform=transform)

# Add the US-CZ2 Flux Tower point
ax.plot(US_CZ2_coords[0], US_CZ2_coords[1], marker='*', color='black', markersize=15, transform=transform)
ax.text(US_CZ2_coords[0] + 0.02, US_CZ2_coords[1], 'Flux Tower', transform=transform, fontsize=18, color='black')
# plot the outline of the polygon
# ax.add_geometries(polygon_gdf.geometry, crs=ccrs.PlateCarree(), edgecolor=line_col, facecolor='none', linewidth=1.5)

# add_latlon_ticks(ax, extent, x_rotation=30, labelsize = 8)
plt.xlabel(None)
plt.ylabel(None)
# plt.title(f'Summer {drought_year}', fontsize = 16)
plt.title('')
plt.show()

#%%
##### PLOT sample plot, just for the colorbar

print(f'plotting overall annual mean map')
# fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': crs}) # establish the figure, axes with larger size
fig, ax = plt.subplots(figsize=(4, 3), subplot_kw={'projection': crs}) # establish the figure, axes

# Set extent of map before adding base img
ax.set_extent(extent, crs=ccrs.PlateCarree())  # PlateCarree is default lat/lon

cbar_kwargs={'orientation': 'vertical',
        'pad': 0.1,
        'label': 'CCI',
        'shrink': 0.75,
        'format': FFmt(z_axis_formatter),
        'aspect': 8,  # Increase the aspect ratio to make the colorbar wider
        'label': ''}  # Increase the font size of the colorbar labels

dat_ds_drought[plot_var].plot(ax=ax, transform=transform, alpha=alpha,
              cmap='viridis_r', vmin = zmin, vmax = zmax,
              cbar_kwargs=cbar_kwargs)
plt.title('')
plt.show()


# %%
