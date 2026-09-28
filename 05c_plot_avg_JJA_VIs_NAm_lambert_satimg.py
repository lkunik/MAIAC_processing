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


# def test_plot(dat_da, plot_title, zlab, plot_extent = None, plot_filename = None, plot_min = None, plot_max = None):
#     if plot_extent is None:
#         plot_extent = get_tile_plot_extent(dat_da)
#     if plot_min is None:
#         plot_min = float(np.nanmin(dat_da))
#     if plot_max is None:
#         plot_max = float(np.nanmax(dat_da))

#     fig, ax = plt.subplots(figsize=(15, 15), subplot_kw={'projection': crs}) # establish the figure, axes
#     ax.set_extent(plot_extent, crs=ccrs.PlateCarree())  # PlateCarree is default lat/lon    
#     ax.add_image(tiler, tiler_zoom, interpolation='none') # Add base image
#     ax.add_feature(states_geom, facecolor="none", edgecolor="black", linewidth=1) # add state boundaries to the map
#     dat_da.plot(ax=ax, x='lon', y='lat', transform=transform, 
#                     alpha=alpha, cmap='inferno_r',
#                     vmin=plot_min, vmax=plot_max,
#                     cbar_kwargs={'orientation': 'vertical',
#                                             'pad': 0.05,
#                                             'label': zlab,
#                                             'shrink': 0.75})

#     plt.title(plot_title, fontsize=18)
#     plt.xlabel('lon', fontsize=15)
#     plt.ylabel('lat', fontsize=14)

#     if plot_filename is not None:
#         if not os.path.exists(os.path.dirname(plot_filename)):
#             Path(os.path.dirname(plot_filename)).mkdir(parents=True, exist_ok=True)
#         plt.savefig(plot_filename, format = "png", bbox_inches = "tight")
#         plt.close('all')
#     else:
#         plt.show()


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
    # 'h08v04',
    # 'h08v05',
	# 'h08v06',
	# 'h08v07', 
    # 'h09v02', 
	# 'h09v03',  
    # 'h09v04',
    # 'h09v05', 
	# 'h09v06',
	# 'h09v07', 
	# 'h09v08',  
	# 'h10v02', 
	# 'h10v03',
    # 'h10v04',
	# 'h10v05',
	# 'h10v06', 
	# 'h10v07',
	# 'h10v08', 
	# 'h11v02', 
	# 'h11v03', 
	# 'h11v04', 
	# 'h11v05', 
    # 'h11v06', 
    # 'h11v07',  
	# 'h12v02',
	# 'h12v03', 
	# 'h12v04',
    # 'h12v05',
	# 'h13v02', 
	# 'h13v03', 
	# 'h13v04',
	# 'h14v02',
    # 'h14v03',
    # 'h14v04',
    # "h12v01",
    # "h13v01",
    # "h14v01",
    # "h15v01",
    # "h15v02"
    ]
QC_option = 'CloudFree' # options are 'CloudAOD', 'Cloud', 'AOD', or 'None'
avg_window = 'JJA'

dat_basedir = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/01d_rect/QCfilt_{QC_option}/'
tile_dirs = [os.path.join(dat_basedir, tile, 'annualized') for tile in MODIS_tiles]


plot_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/plots/maps/North_America/'


ESA_CCI_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/ESA_CCI/LandCover/'

# Sample 0.01° 1D Rectilinear file for North America
ESA_CCI_file = os.path.join(ESA_CCI_dir, f'forest_fraction_0.01deg_north_america.nc')
ESA_CCI_xr = xr.open_dataset(ESA_CCI_file, decode_coords = 'all').rename({'x': 'lon', 'y': 'lat'}).sortby(['lon', 'lat'], ascending = True)['fraction_forest']
ESA_CCI_xr.rio.write_crs('EPSG:4326', inplace=True)


#%%

start_time = time.time()
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

overall_CCI_min = -0.3658804343909615
overall_CCI_max = 0.2306400408606525
overall_NDVI_min = 0.019227114090525473
overall_NDVI_max = 0.9046409437773152
#%%

zlab = 'MCD19 CCI'
plot_year = 'avg'
# Set up Lambert figure
projection = ccrs.LambertConformal(central_longitude=-100, central_latitude=40)
fig, ax = plt.subplots(figsize=(20, 20), subplot_kw={'projection': projection})

ax.stock_img()
ax.set_extent([-170, -50, 9, 72], crs=ccrs.PlateCarree())
# ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.7)
ax.add_feature(cfeature.OCEAN, facecolor='black', alpha=0.55, zorder = 2)
ax.add_feature(cfeature.COASTLINE, edgecolor='black', zorder = 3)
# lakes_gdf.plot(ax=ax, facecolor='#14629c', edgecolor="#0b4a7a", linewidth=1, alpha=0.7)
lakes_lambert = lakes_gdf.to_crs(projection.proj4_init)
lakes_lambert.plot(ax=ax, facecolor='#73ADFF', edgecolor="#0b4a7a", linewidth=1, alpha=0.7)

for itile, tile_dir in enumerate(tile_dirs):
    print(f'Plotting CCI for tile {MODIS_tiles[itile]}...')
    MCD19A1_file = os.path.join(tile_dir, f'MCD19A1.{MODIS_tiles[itile]}.061_01d_annual_{avg_window}_CV-MVC_QCfilt_{QC_option}.nc')

    if not os.path.exists(MCD19A1_file):
        print(f'File not found: {MCD19A1_file}. Skipping this tile.')
        continue
    dat_ds = xr.open_dataset(MCD19A1_file).drop_vars(['NDVI'])
    dat_ds = dat_ds.assign(CCI=(dat_ds["Sur_refl11"] - dat_ds["Sur_refl1"])/
            (dat_ds["Sur_refl11"] + dat_ds["Sur_refl1"]))

    dat_ds_CCI_mean = dat_ds['CCI'].mean(dim='year', skipna=True)

    # # if itile < (len(tile_dirs)-1):
    # dat_ds_CCI_mean.plot(ax=ax, x='x', y='y', transform=ccrs.PlateCarree(), 
    #                 alpha=1, cmap='YlGn',
    #                 vmin=overall_CCI_min, vmax=overall_CCI_max,
    #                 add_colorbar=False)
    # Place color bar inside plot area at (0.3, 0.4)
    im = dat_ds_CCI_mean.plot(
        ax=ax, x='x', y='y', transform=ccrs.PlateCarree(),
        alpha=1, cmap='YlGn',
        vmin=overall_CCI_min, vmax=overall_CCI_max,
        add_colorbar=False
    )


    # Add a gray box behind the colorbar
    cbar_box = plt.Rectangle(
        (0.22, 0.28), 0.22, 0.10,  # [left, bottom], width, height (slightly larger than cbar_ax)
        transform=fig.transFigure,
        color='#e0e0e0', alpha=0.8, zorder=3, linewidth=0
    )
    fig.patches.append(cbar_box)


    cbar_ax = fig.add_axes([0.23, 0.31, 0.2, 0.04])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar_ax.set_zorder(4)  # Set zorder on the axes, not the colorbar
    # Move colorbar label to the top, keep ticks at the bottom
    cbar_ax.xaxis.set_label_position('top')
    cbar_ax.xaxis.set_ticks_position('bottom')
    cbar.set_label(f'{zlab} - {plot_year}', fontsize=23, labelpad=10)
    cbar.ax.tick_params(labelsize=23)
    cbar.set_ticks([-0.3, 0.2])
    


ax.add_feature(states_geom, facecolor="none", edgecolor="#737373", linewidth=0.4) # add state boundaries to the map
ax.add_feature(canada_geom, facecolor="none", edgecolor="#737373", linewidth=0.4) # add Canadian province boundaries to the map
ax.add_feature(mexico_geom, facecolor="none", edgecolor="#737373", linewidth=0.4) # add Mexican state boundaries to the map
ax.set_title('')
outfile = os.path.join(plot_dir, f'MCD19A1_NorthAmerica_annual_{avg_window}_CCI_map_QCfilt_{QC_option}_lambert.png')
plt.savefig(outfile, dpi=150, bbox_inches='tight')
plt.close()

print(f'CCI map complete. Now plotting NDVI...')

cci_end_time = time.time()
cci_elapsed = cci_end_time - start_time
print(f'CCI plotting took {cci_elapsed:.2f} seconds ({td_min(timedelta(seconds=cci_elapsed))} minutes and {td_sec(timedelta(seconds=cci_elapsed))} seconds).')

#%%

zlab = 'MCD19 NDVI'

# Set up Lambert figure
projection = ccrs.LambertConformal(central_longitude=-100, central_latitude=40)
fig, ax = plt.subplots(figsize=(20, 20), subplot_kw={'projection': projection})

ax.stock_img()
ax.set_extent([-170, -50, 9, 72], crs=ccrs.PlateCarree())
# ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.7)
ax.add_feature(cfeature.OCEAN, facecolor='black', alpha=0.55, zorder = 2)
ax.add_feature(cfeature.COASTLINE, edgecolor='black', zorder = 3)
# lakes_gdf.plot(ax=ax, facecolor='#14629c', edgecolor="#0b4a7a", linewidth=1, alpha=0.7)
lakes_lambert = lakes_gdf.to_crs(projection.proj4_init)
lakes_lambert.plot(ax=ax, facecolor='#73ADFF', edgecolor="#0b4a7a", linewidth=1, alpha=0.7)

for itile, tile_dir in enumerate(tile_dirs):
    print(f'Plotting NDVI for tile {MODIS_tiles[itile]}...')
    MCD19A1_file = os.path.join(tile_dir, f'MCD19A1.{MODIS_tiles[itile]}.061_01d_annual_{avg_window}_CV-MVC_QCfilt_{QC_option}.nc')

    if not os.path.exists(MCD19A1_file):
        print(f'File not found: {MCD19A1_file}. Skipping this tile.')
        continue
    dat_ds = xr.open_dataset(MCD19A1_file).drop_vars(['NDVI'])
    dat_ds = dat_ds.assign(NDVI=(dat_ds["Sur_refl2"] - dat_ds["Sur_refl1"])/
            (dat_ds["Sur_refl2"] + dat_ds["Sur_refl1"]))

    dat_ds_NDVI_mean = dat_ds['NDVI'].mean(dim='year', skipna=True)

    im = dat_ds_NDVI_mean.plot(
        ax=ax, x='x', y='y', transform=ccrs.PlateCarree(),
        alpha=1, cmap='YlGn',
        vmin=overall_NDVI_min, vmax=overall_NDVI_max,
        add_colorbar=False
    )

    # Add a gray box behind the colorbar
    cbar_box = plt.Rectangle(
        (0.22, 0.28), 0.22, 0.10,  # [left, bottom], width, height (slightly larger than cbar_ax)
        transform=fig.transFigure,
        color='#e0e0e0', alpha=0.8, zorder=3, linewidth=0
    )
    fig.patches.append(cbar_box)


    cbar_ax = fig.add_axes([0.23, 0.31, 0.2, 0.04])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar_ax.set_zorder(4)  # Set zorder on the axes, not the colorbar
    # Move colorbar label to the top, keep ticks at the bottom
    cbar_ax.xaxis.set_label_position('top')
    cbar_ax.xaxis.set_ticks_position('bottom')
    cbar.set_label(f'{zlab} - {plot_year}', fontsize=23, labelpad=10)
    cbar.ax.tick_params(labelsize=23)
    cbar.set_ticks([0.1, 0.5, 0.9])

ax.add_feature(states_geom, facecolor="none", edgecolor="#737373", linewidth=0.4) # add state boundaries to the map
ax.add_feature(canada_geom, facecolor="none", edgecolor="#737373", linewidth=0.4) # add Canadian province boundaries to the map
ax.add_feature(mexico_geom, facecolor="none", edgecolor="#737373", linewidth=0.4) # add Mexican state boundaries to the map

outfile = os.path.join(plot_dir, f'MCD19A1_NorthAmerica_annual_{avg_window}_NDVI_map_QCfilt_{QC_option}_lambert.png')
plt.savefig(outfile, dpi=150, bbox_inches='tight')
plt.close()

print(f'NDVI map complete. Now plotting normalized difference...')

ndvi_end_time = time.time()
ndvi_elapsed = ndvi_end_time - start_time
print(f'NDVI plotting took {ndvi_elapsed:.2f} seconds ({td_min(timedelta(seconds=ndvi_elapsed))} minutes and {td_sec(timedelta(seconds=ndvi_elapsed))} seconds).')

# #%%


# overall_normdiff_min = 1
# overall_normdiff_max = -1
# print('Calculating normalized difference and getting min and max...')
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

#     dat_ds_CCI_mean_scaled = (dat_ds_CCI_mean - overall_CCI_min) / (overall_CCI_max - overall_CCI_min)
#     dat_ds_NDVI_mean_scaled = (dat_ds_NDVI_mean - overall_NDVI_min) / (overall_NDVI_max - overall_NDVI_min)
#     dat_ds_norm_diff = dat_ds_CCI_mean_scaled - dat_ds_NDVI_mean_scaled

#     overall_normdiff_min = min(overall_normdiff_min, np.nanquantile(dat_ds_norm_diff.values, 0.01))
#     overall_normdiff_max = max(overall_normdiff_max, np.nanquantile(dat_ds_norm_diff.values, 0.99))


# plot_normdiff_min = -max(abs(overall_normdiff_min), abs(overall_normdiff_max)) # -1 #
# plot_normdiff_max = max(abs(overall_normdiff_min), abs(overall_normdiff_max)) # 1 #

# zlab = 'CCI-NDVI norm'

# # Set up Lambert figure
# projection = ccrs.LambertConformal(central_longitude=-100, central_latitude=40)
# fig, ax = plt.subplots(figsize=(20, 20), subplot_kw={'projection': projection})

# # ax.set_extent(plot_extent, crs=ccrs.PlateCarree())  # PlateCarree is default lat/lon 
# ax.set_extent([-170, -50, 9, 72], crs=ccrs.PlateCarree())
# ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.7)
# ax.add_feature(cfeature.OCEAN, facecolor='black', alpha=0.9)
# ax.add_feature(cfeature.COASTLINE, edgecolor='black')
# lakes_gdf.plot(ax=ax, facecolor='#2b8dd6', edgecolor='#14629c', linewidth=1, alpha=0.7)

# for itile, tile_dir in enumerate(tile_dirs):
#     print(f'Plotting normalized difference for tile {MODIS_tiles[itile]}...')
#     MCD19A1_file = os.path.join(tile_dir, f'MCD19A1.{MODIS_tiles[itile]}.061_01d_annual_{avg_window}_CV-MVC_QCfilt_{QC_option}.nc')

#     if not os.path.exists(MCD19A1_file):
#         print(f'File not found: {MCD19A1_file}. Skipping this tile.')
#         continue
#     dat_ds = xr.open_dataset(MCD19A1_file).drop_vars(['NDVI'])
#     dat_ds = dat_ds.assign(CCI=(dat_ds["Sur_refl11"] - dat_ds["Sur_refl1"])/
#             (dat_ds["Sur_refl11"] + dat_ds["Sur_refl1"]))
#     dat_ds = dat_ds.assign(NDVI=(dat_ds["Sur_refl2"] - dat_ds["Sur_refl1"])/
#             (dat_ds["Sur_refl2"] + dat_ds["Sur_refl1"]))

#     dat_ds_CCI_mean = dat_ds['CCI'].mean(dim='year', skipna=True)
#     dat_ds_NDVI_mean = dat_ds['NDVI'].mean(dim='year', skipna=True)

#     dat_ds_CCI_mean_scaled = (dat_ds_CCI_mean - overall_CCI_min) / (overall_CCI_max - overall_CCI_min)
#     dat_ds_NDVI_mean_scaled = (dat_ds_NDVI_mean - overall_NDVI_min) / (overall_NDVI_max - overall_NDVI_min)
#     dat_ds_norm_diff = dat_ds_CCI_mean_scaled - dat_ds_NDVI_mean_scaled

#     # if itile < (len(tile_dirs)-1):
#     dat_ds_norm_diff.plot(ax=ax, x='x', y='y', transform=ccrs.PlateCarree(), 
#                     alpha=1, cmap='PRGn',
#                     vmin=plot_normdiff_min, vmax=plot_normdiff_max,
#                     add_colorbar=False)
#     # else:
#     #     dat_ds_norm_diff.plot(ax=ax, x='x', y='y', transform=ccrs.PlateCarree(), 
#     #                 alpha=1, cmap='PRGn',
#     #                 vmin=plot_normdiff_min, vmax=plot_normdiff_max,
#     #                 cbar_kwargs={'orientation': 'vertical',
#     #                                     'pad': 0.05,
#     #                                     'label': zlab,
#     #                                     'shrink': 0.6})

# ax.add_feature(states_geom, facecolor="none", edgecolor="#737373", linewidth=0.4) # add state boundaries to the map
# ax.add_feature(canada_geom, facecolor="none", edgecolor="#737373", linewidth=0.4) # add Canadian province boundaries to the map
# ax.add_feature(mexico_geom, facecolor="none", edgecolor="#737373", linewidth=0.4) # add Mexican state boundaries to the map

# outfile = os.path.join(plot_dir, f'MCD19A1_NorthAmerica_annual_{avg_window}_CCI-NDVInorm_map_QCfilt_{QC_option}_lambert.png')
# plt.savefig(outfile, dpi=150, bbox_inches='tight')
# plt.close()

# print(f'CCI-NDVI normalized difference map complete.')




# %%
