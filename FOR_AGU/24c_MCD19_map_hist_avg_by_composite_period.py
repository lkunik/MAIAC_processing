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
import rasterio
from rasterio.enums import Resampling
import numpy as np
import rasterio
from rasterio.warp import reproject
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
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

QC_option = 'Cloud'
MCD19_dir = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/01d_rect/QCfilt_{QC_option}/WUS_combined/hist_avg/'
MCD19_files = sorted(glob.glob(f'{MCD19_dir}/*.nc'))

plot_dir = f'/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/plots/maps/seasonality/QCfilt_{QC_option}'
Path(plot_dir).mkdir(parents=True, exist_ok=True)

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

tiler_zoom = 6  # define a zoom level of detail
plot_extent = [-125.3, -102.7, 29.7, 50.3]  # [minx, maxx, miny, maxy], bounds for map

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

analysis_years = np.arange(2000, 2025)
DOY_composite_starts = np.arange(1, 366, 16)
composite_dates_2020 = [dt(2020, 1, 1) + timedelta(days=int(DOY) - 1) for DOY in DOY_composite_starts]

#########################################
# Begin main
#########################################
#%%

plot_var = 'CCI'
zax_label = plot_var

plot_min = 1
plot_max = 0
for ii, MCD19_file in enumerate(MCD19_files):
    print(f'Checking dataset bounds for composite period {ii:02d}...')
    MCD19_xr = xr.open_dataset(MCD19_file, decode_coords = 'all')
    file_p1 = np.nanpercentile(MCD19_xr[plot_var].values, 1)
    file_p99 = np.nanpercentile(MCD19_xr[plot_var].values, 99)
    plot_min = min(plot_min, file_p1)
    plot_max = max(plot_max, file_p99)


for ii, MCD19_file in enumerate(MCD19_files):
    print(f'Plotting composite period {ii:02d}...')
    MCD19_xr = xr.open_dataset(MCD19_file, decode_coords = 'all')

    mm = composite_dates_2020[ii].month
    dd = composite_dates_2020[ii].day
    date_string = f'{mm:02d}-{dd:02d}'

    fig, ax = plt.subplots(subplot_kw={'projection': crs}, figsize=(8, 7)) # establish the figure, axes

    # Set extent of map before adding base img
    ax.set_extent(plot_extent, crs=ccrs.PlateCarree())  # PlateCarree is default lat/lon
    ax.add_image(tiler, tiler_zoom, interpolation='none', alpha=0.9) # Add base image

    alpha = 0.85  # transparency 0-1
    mappable = MCD19_xr[plot_var].plot(ax=ax, 
                transform=transform, alpha=alpha, 
                cmap='YlGn', add_colorbar=False,
                vmin=plot_min, vmax=plot_max)


    cbar = plt.colorbar(mappable, ax=ax, orientation='vertical', pad=-0.02, shrink=0.5, format=FFmt(z_axis_formatter))
    cbar.ax.set_position([0.82, 0.25, 0.05, 0.5])  # [left, bottom, width, height]
    # cbar.set_ticks([-0.3, -0.15, 0, 0.15, 0.3])
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label(zax_label, fontsize=23)

    # add state boundaries to the map
    ax.add_feature(states_geom, facecolor="none", edgecolor="black", linewidth=1)

    plt.title(f'MCD19A1 {plot_var} - Period {ii:02d} mean ({date_string})', fontsize=20)
    plt.xlabel('', fontsize=10)
    plt.ylabel('', fontsize=10)

    mapfile = f'map_MCD19A1_{plot_var}_WUS_pd{ii:02d}.png'
    plt.savefig(os.path.join(plot_dir, mapfile), format = 'png', bbox_inches="tight")
    plt.close('all') # close the plot

