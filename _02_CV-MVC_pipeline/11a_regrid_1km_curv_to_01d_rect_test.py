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


#########################################
# Define Global functions
#########################################


####################################
### Set up map parameters
####################################

# use Google Satellite imagery as basemap
tiler = img_tiles.GoogleTiles(style='satellite')
crs = tiler.crs # set crs of map tiler
alpha = 0.5  # transparency 0-1
transform = ccrs.PlateCarree()  # transform specifies the crs that the data is in
plot_extent = [-128, -106, 39, 51]  # [minx, maxx, miny, maxy], bounds for western US

### Load state boundaries
fn = shpreader.natural_earth(
   resolution='10m', category='cultural', 
   name='admin_1_states_provinces',
)

reader = shpreader.Reader(fn)
states = [x for x in reader.records() if x.attributes["admin"] == "United States of America"] # get all states in US
states_geom = cfeature.ShapelyFeature([x.geometry for x in states], ccrs.PlateCarree())



tiler_zoom = 5  # define a zoom level of detail


tile_plot_extents = {
    'h08v04': [-125, -120, 39, 45],
    'h08v05': [-125, -106, 39, 41],
    'h09v04': [-128, -106, 39, 51],
    'h09v05': [-120, -106, 29, 41],
    'h10v04': [-125, -100, 39, 51],
    # Add more tiles and their extents as needed
}


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



def format_elapsed(start_time):
    """Elapsed time since start_time (from time.time()) as e.g. '0h 3m 25.4s'."""
    elapsed = time.time() - start_time
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    return f'{int(hours)}h {int(minutes)}m {seconds:.1f}s'


def regrid_flag_nearest(regridder, flag_da, ref_da):
    """Nearest-neighbour regridding of a categorical variable (e.g. QA_flag), reusing the weights of a bilinear
    xESMF regridder instead of building a separate (slow) nearest-neighbour regridder: each destination cell takes
    the value of the source pixel with the largest bilinear weight, i.e. the nearest of the surrounding source pixels.
    flag_da: 2D source variable on the regridder's input grid
    ref_da: a bilinear-regridded variable (e.g. Sur_refl1) on the output grid, used for the output coordinates and
            data footprint: cells with no data in ref_da are set to 0 (no data); cells outside the source tile are NaN"""
    in_dims = flag_da['lat'].dims                          # horizontal dims of the source grid, e.g. ('y', 'x')
    src = flag_da.transpose(*in_dims).values
    if src.shape != tuple(regridder.shape_in):
        raise ValueError(f'{flag_da.name} shape {src.shape} does not match the regridder input grid {regridder.shape_in}')
    if ref_da.shape != tuple(regridder.shape_out):
        raise ValueError(f'reference shape {ref_da.shape} does not match the regridder output grid {regridder.shape_out}')

    W = regridder.weights.data.tocsr()                     # sparse (n_out, n_in) bilinear weights
    W.data = np.nan_to_num(W.data, nan=0.0)                # unmapped_to_nan puts a NaN placeholder in rows with no source pixels
    W.eliminate_zeros()
    mapped = np.diff(W.indptr) > 0                         # destination cells inside the source footprint
    nn_idx = np.asarray(W.argmax(axis=1)).ravel()          # source pixel with the largest weight for each destination cell
    flag_out = np.where(mapped, src.ravel()[nn_idx], np.nan).reshape(regridder.shape_out)
    flag_out = np.where(mapped.reshape(regridder.shape_out) & np.isnan(ref_da.values), 0, flag_out)   # no data after bilinear regridding -> 0

    return xr.DataArray(flag_out, dims=ref_da.dims, coords=ref_da.coords, attrs=flag_da.attrs)


#########################################
# Define Global Variables and constants
#########################################

plot_years = [2000, 2005, 2010, 2015, 2020, 2025]
plot_period = 12 # 16-day period to plot
#########################################
# Begin main
#########################################

#%%
# def main():
#     try:
MODIS_tile = 'h11v03' # sys.argv[1] #
composite_year = 2012 #int(sys.argv[2]) # command line argument
composite_period = 14 #int(sys.argv[3]) # command line argument
QC_option = 'CloudFree_LowAOD_ClearAdj' #sys.argv[4] # command line argument - options are 'CloudFree', 'CloudAOD', 'Cloud', 'AOD', or 'None'

dat_basedir = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/lm-fill/QCfilt_{QC_option}/{MODIS_tile}'
out_dir = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/01d_rect/QCfilt_{QC_option}/{MODIS_tile}'
plot_dir = f'/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/plots/maps/regrid_validation/QCfilt_{QC_option}/{MODIS_tile}'
Path(plot_dir).mkdir(parents=True, exist_ok=True)

# Sample filename: MCD19A1.h14v03.061_1km_16day_CV-MVC_lmFill_QCfilt_CloudFree_2025-13.nc
MCD19A1_file = os.path.join(dat_basedir, f'MCD19A1.{MODIS_tile}.061_1km_16day_CV-MVC_lmFill_QCfilt_{QC_option}_{str(composite_year)}-{composite_period:02d}.nc')
if not os.path.exists(MCD19A1_file):
    print(f"No MCD19A1 file found for tile {MODIS_tile}, composite_year {composite_year}, composite_period {composite_period}, QC_option {QC_option}, exiting with error code...")
    # sys.exit(1)

out_file = os.path.join(out_dir, f'MCD19A1.{MODIS_tile}.061_01d_16day_CV-MVC_QCfilt_{QC_option}_{composite_year}-{composite_period:02d}_rect.nc')
if os.path.exists(out_file):
    print(f"Output file already exists, exiting...")
    # sys.exit(0)

ESA_CCI_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/ESA_CCI/LandCover/'

# Get the tile corners from the metadata
tile_corners = MODIS_tilecorners_df.loc[MODIS_tile]


# Load in file to establish regridder
dat_ds_1km_curv = xr.open_dataset(MCD19A1_file)

# Sample 0.01° 1D Rectilinear file for North America
ESA_CCI_file = os.path.join(ESA_CCI_dir, f'forest_fraction_0.01deg_north_america_FULL.nc')
ESA_CCI_xr = xr.open_dataset(ESA_CCI_file, decode_coords = 'all').rename({'x': 'lon', 'y': 'lat'}).sortby(['lon', 'lat'], ascending = True)['fraction_forest']
ESA_CCI_xr.rio.write_crs('EPSG:4326', inplace=True)
ESA_CCI_xr_tile = ESA_CCI_xr.sel(lat=slice(tile_corners.lat_min-0.02, tile_corners.lat_max+0.02), lon=slice(tile_corners.lon_min-0.02, tile_corners.lon_max+0.02))

#%%


# Use XESMF package to establish a regridder which we can recycle
t_start = time.time()
regridder = xe.Regridder(dat_ds_1km_curv, ESA_CCI_xr_tile, "bilinear", unmapped_to_nan = True)
print(f'Bilinear regridder built in {format_elapsed(t_start)}', flush=True)
# Regrid to 0.01 deg rectilinear grid (QA_flag is categorical, so it is regridded by nearest neighbour below)
dat_ds_01d_rect = regridder(dat_ds_1km_curv.drop_vars('QA_flag', errors='ignore'))
# QA_flag: nearest neighbour, reusing the bilinear weights (a separate "nearest_s2d" regridder is very slow to build
# and extrapolates the tile-edge flags to every cell of the lat/lon box outside the tile)
if 'QA_flag' in dat_ds_1km_curv:
    t_start = time.time()
    dat_ds_01d_rect['QA_flag'] = regrid_flag_nearest(regridder, dat_ds_1km_curv['QA_flag'], dat_ds_01d_rect['Sur_refl1'])
    print(f'QA_flag nearest-neighbour regridding done in {format_elapsed(t_start)}', flush=True)
# Assign time coordinate from original dataset
dat_ds_01d_rect = dat_ds_01d_rect.assign_coords(time=dat_ds_1km_curv.time.values)


if not os.path.exists(os.path.dirname(out_file)):
    Path(os.path.dirname(out_file)).mkdir(parents=True, exist_ok=True)
dat_ds_01d_rect.to_netcdf(out_file)

### Plot a sample to verify that regridding worked
if (int(composite_year) in plot_years) and (int(composite_period) == plot_period):

    tile_plot_extent = tile_plot_extents[MODIS_tile]
    # plot original 1km curvilinear data
    plot_file = os.path.join(plot_dir, f'MCD19A1_{MODIS_tile}_1km_curv_{composite_year}-{composite_period}.png')
    test_plot(dat_ds_1km_curv['Sur_refl1'],plot_title=f'1km curvilinear, {composite_year}-{composite_period}', 
            zlab='Refl', plot_filename=plot_file, plot_extent=tile_plot_extent)

    # plot regridded 0.01 deg rectilinear data (rectilinear grid)
    plot_file_rect = os.path.join(plot_dir, f'MCD19A1_{MODIS_tile}_01d_rect_{composite_year}-{composite_period}.png')
    test_plot(dat_ds_01d_rect['Sur_refl1'], plot_title=f'0.01° Rectilinear, {composite_year}-{composite_period}', 
            zlab='Refl', plot_filename=plot_file_rect, plot_extent=tile_plot_extent)

# Clean up
dat_ds_1km_curv.close()
dat_ds_01d_rect.close()
gc.collect()

    # except Exception as e:
    #     print(f"ERROR: {str(e)}")
    #     import traceback
    #     traceback.print_exc()
    #     sys.exit(1)  
    
# if __name__ == "__main__":
#     main()

# %%
