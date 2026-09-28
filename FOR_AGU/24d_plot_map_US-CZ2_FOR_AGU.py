#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Lewis Kunik - University of Utah
# Contact: lewis.kunik@utah.edu
#
#

#%%
#########################################
# Load packages
#########################################

import os  # operating system library
import sys
from pathlib import Path

# Custom packages
utils_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/pyutil'
if utils_dir not in sys.path:
   sys.path.append(utils_dir)
import data_utils
import plot_utils
import MODIS_tile_corners as MOD_ref


# runtime packages
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# numerical/data management packages
import numpy as np
import xarray as xr  # for multi dimensional data
import geopandas as gpd
# time and date packages
import time

# plotting packages
from matplotlib import pyplot as plt  # primary plotting module
import cartopy.crs as ccrs  # spatial plotting library 
from cartopy.io import img_tiles  # cartopy's implementation of webtiles
import cartopy.io.shapereader as shpreader
import cartopy.feature as cfeature
from matplotlib.ticker import FuncFormatter as FFmt  # need this for formatting plot ticks
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from shapely.geometry import mapping
from shapely.geometry import Polygon
import rasterio
from matplotlib.patches import Polygon as MplPolygon

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
### Set up plotting parameters
####################################

plt.rcParams.update({
   'axes.labelsize': 16,
   'axes.titlesize': 16,
   'xtick.labelsize': 16,
   'ytick.labelsize': 15,
   'legend.fontsize': 13,
})

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

# matplotlib map plotting functions
def z_axis_formatter(x, pos, deci=2):
   '''Format ticks to format with deci number of decimals'''

   return f'{x:.{deci}f}'



#########################################
# Define Global Variables and constants
#########################################

#########################################
# Begin main
#########################################


#%%

plot_extent = [-119.7, -117.9, 36.4, 37.8]  # [minx, maxx, miny, maxy], bounds for map
tiler_zoom = 10  # define a zoom level of detail

CZ2_lon = -119.2566
CZ2_lat = 37.0311 

fig, ax = plt.subplots(subplot_kw={'projection': crs}, figsize=(15, 12)) # establish the figure, axes

# Set extent of map before adding base img
ax.set_extent(plot_extent, crs=ccrs.PlateCarree())  # PlateCarree is default lat/lon
ax.add_image(tiler, tiler_zoom, interpolation='none', alpha=0.9) # Add base image

# Plot a red star marker at the CZ2 location
ax.plot(CZ2_lon, CZ2_lat, marker='*', color='yellow', markersize=60, markeredgecolor='black', transform=ccrs.PlateCarree(), zorder=10)
# add state boundaries to the map
ax.add_feature(states_geom, facecolor="none", edgecolor="black", linewidth=1)

plt.title('')
plt.xlabel('', fontsize=10)
plt.ylabel('', fontsize=10)

plt.show()
# plt.show()


# %%



WUS_extent = [-125.3, -112, 32, 42.5]  # [minx, maxx, miny, maxy], bounds for map
tiler_zoom = 7  # define a zoom level of detail

fig, ax = plt.subplots(subplot_kw={'projection': crs}, figsize=(15, 12)) # establish the figure, axes

# Set extent of map before adding base img
ax.set_extent(WUS_extent, crs=ccrs.PlateCarree())  # PlateCarree is default lat/lon
ax.add_image(tiler, tiler_zoom, interpolation='none', alpha=0.9) # Add base image
# Define the vertices of the box using plot_extent
box_coords = [
    [plot_extent[0], plot_extent[2]],  # lower left
    [plot_extent[1], plot_extent[2]],  # lower right
    [plot_extent[1], plot_extent[3]],  # upper right
    [plot_extent[0], plot_extent[3]],  # upper left
    [plot_extent[0], plot_extent[2]],  # close the box
]

# Create the polygon patch
box_patch = MplPolygon(
    box_coords,
    closed=True,
    edgecolor='darkred',
    facecolor='none',
    alpha=1,
    linewidth=2.5,
    transform=ccrs.PlateCarree(),
    zorder=5
)

ax.add_patch(box_patch)

# Create the polygon patch
box_fill_patch = MplPolygon(
    box_coords,
    closed=True,
    edgecolor='none',
    facecolor='red',
    alpha=0.2,
    linewidth=2,
    transform=ccrs.PlateCarree(),
    zorder=5
)

ax.add_patch(box_fill_patch)
ax.add_feature(states_geom, facecolor="none", edgecolor="black", linewidth=2.5)

plt.title('')
plt.xlabel('', fontsize=10)
plt.ylabel('', fontsize=10)

plt.show()
# plt.show()


# %%
