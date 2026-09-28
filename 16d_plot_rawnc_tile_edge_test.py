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

# Add seasonal progress indicator (pie chart showing time elapsed)
from matplotlib.patches import Wedge
import matplotlib.patches as mpatches

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

arccos_v = np.vectorize(np.arccos)

def rad_to_deg(rad):
   return rad*180/math.pi

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
    # 'h07v06',  
    # 'h08v04',
    # 'h08v05',
	# 'h08v06',
	# 'h08v07', 
    # 'h09v02', 
	# 'h09v03',  
    'h09v04',
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

# QC_option = 'CloudFree' # options are 'CloudAOD', 'Cloud', 'AOD', or 'None'
# avg_window = 'JJA'

dat_basedir = '/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/pre_proc/raw_nc/'
# tile_dirs = [os.path.join(dat_basedir, tile) for tile in MODIS_tiles]

plot_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/plots/maps/regrid_validation/tile_edge_test/'
Path(plot_dir).mkdir(parents=True, exist_ok=True)

DOY_composite_starts = np.arange(1, 366, 16)
composite_dates_2020 = [dt(2020, 1, 1) + timedelta(days=int(DOY) - 1) for DOY in DOY_composite_starts]

composite_periods = np.arange(0,23)
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

#     dat_ds_CCI_composite = dat_ds['CCI'].mean(dim='year', skipna=True)
#     dat_ds_NDVI_composite = dat_ds['NDVI'].mean(dim='year', skipna=True)

#     overall_CCI_min = float(np.min([np.nanquantile(dat_ds_CCI_composite.values, 0.01), overall_CCI_min]))
#     overall_CCI_max = float(np.max([np.nanquantile(dat_ds_CCI_composite.values, 0.99), overall_CCI_max]))
#     overall_NDVI_min = float(np.min([np.nanquantile(dat_ds_NDVI_composite.values, 0.01), overall_NDVI_min]))
#     overall_NDVI_max = float(np.max([np.nanquantile(dat_ds_NDVI_composite.values, 0.99), overall_NDVI_max]))

# print(f'Overall CCI min: {overall_CCI_min}, Overall CCI max: {overall_CCI_max}')
# print(f'Overall NDVI min: {overall_NDVI_min}, Overall NDVI max: {overall_NDVI_max}')
# Overall CCI min: -0.3658804343909615, Overall CCI max: 0.2306400408606525
# Overall NDVI min: 0.019227114090525473, Overall NDVI max: 0.9046409437773152

overall_CCI_min = -0.37
overall_CCI_max = 0.25
overall_NDVI_min = 0.015
overall_NDVI_max = 0.95
#%%

composite_years = np.arange(2011, 2018)
composite_periods = np.arange(0, 23)

# def main():
zlab = 'MCD19 VZA (°)'

composite_year = 2025 # get the composite year to plot from the command line argument
plot_doy = 225

print(f'Plotting map for DOY {plot_doy} of year {composite_year}...')
# Set up Lambert figure
projection = ccrs.LambertConformal(central_longitude=-100, central_latitude=40)

start_period_time = time.time()

nc_vars1km =["Orbits", "lon", "lat", "Sur_refl1", "Sur_refl2", "Sur_refl4", "Sur_refl6", "Sur_refl7", "Sur_refl11", "Status_QA"]
nc_vars5km =["Orbits", "lon", "lat", "cosVZA", "cosSZA", "RelAZ", "Scattering_Angle", "Glint_Angle"]
cosVZA_scale = 0.0001
cosVZA_valid_min = 0 * cosVZA_scale
cosVZA_valid_max = 10000 * cosVZA_scale


#%%



MODIS_tile = 'h09v04'
for itile, MODIS_tile in enumerate(MODIS_tiles):
    # print(f'Plotting VZA for tile {MODIS_tiles[itile]}...')

    # Sample file: MCD19A1.A2025244.h09v04.061.2025245221826.nc
    MCD19A1_files = glob.glob(f'{dat_basedir}MCD19A1.A{composite_year}{plot_doy}.{MODIS_tile}.061.*.nc')

    if not MCD19A1_files:
        print(f'File not found for tile {MODIS_tile}. Skipping this tile.')
        continue
    MCD19A1_file = MCD19A1_files[0]
    dat_ds_1km = xr.open_dataset(MCD19A1_file, group = "grid1km/Data_Fields")[nc_vars1km]
    dat_ds_1km = dat_ds_1km.rename({'XDim': 'x', 'YDim': 'y'})
    dat_ds_1km.rio.write_crs(4326, inplace = True) # specify WGS84 as CRS
    dat_ds_5km = xr.open_dataset(MCD19A1_file, group = "grid5km/Data_Fields")[nc_vars5km]
    dat_ds_5km = dat_ds_5km.rename({'XDim': 'x', 'YDim': 'y'})
    dat_ds_5km.rio.write_crs(4326, inplace = True) # specify WGS84 as CRS
    dat_ds_5km = dat_ds_5km.where((dat_ds_5km['cosVZA'] >= cosVZA_valid_min) & (dat_ds_5km['cosVZA'] < cosVZA_valid_max))

    regridder_5km_to_1km = xe.Regridder(dat_ds_5km, dat_ds_1km, 'bilinear', extrap_method = 'nearest_s2d')
    dat_ds_5km_interp = regridder_5km_to_1km(dat_ds_5km)

    for iorbit in range(dat_ds_1km.coords['Orbits'].size):
        dat_ds_orbit = dat_ds_1km.isel(Orbits=iorbit)
        dat_ds_5km_orbit_interp = dat_ds_5km_interp.isel(Orbits=iorbit)
        dat_ds_5km_orbit = dat_ds_5km.isel(Orbits=iorbit)

        fig, ax = plt.subplots(figsize=(20, 20), subplot_kw={'projection': projection})

        ax.stock_img()
        ax.set_extent([-107, -103, 39, 41], crs=ccrs.PlateCarree())
        # ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.7, zorder = 2)
        # ax.add_feature(cfeature.OCEAN, facecolor='black', alpha=0.55, zorder = 2)
        # ax.add_feature(cfeature.COASTLINE, edgecolor='black', zorder = 3)
        # lakes_lambert = lakes_gdf.to_crs(projection.proj4_init)
        # lakes_lambert.plot(ax=ax, facecolor='#73ADFF', edgecolor="#0b4a7a", linewidth=1, alpha=0.7, zorder = 3)

        im = dat_ds_5km_orbit_interp['cosVZA'].plot(
            ax=ax, x='lon', y='lat', transform=ccrs.PlateCarree(),
            alpha=1, cmap='hot',
            vmin=0, vmax=1,
            add_colorbar=False, zorder = 4
        )

        # Add a gray box behind the colorbar
        cbar_box = plt.Rectangle(
            (0.22, 0.28), 0.22, 0.12,  # [left, bottom], width, height (slightly larger than cbar_ax)
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
        cbar.set_label(f'{zlab}', fontsize=23, labelpad=10)
        cbar.ax.tick_params(labelsize=23)
        cbar.set_ticks([0, 1])
        
        ax.set_title('')
        outfile = os.path.join(plot_dir, f'MCD19A1_tile_edge_cosVZA_1kmregrid_map_rawnc_{composite_year:04d}-{plot_doy:03d}_orbit_{iorbit:02d}.png')
        plt.savefig(outfile, dpi=100, bbox_inches='tight')
        plt.close()
        print('plot saved to: ', os.path.basename(outfile))
    

        fig, ax = plt.subplots(figsize=(20, 20), subplot_kw={'projection': projection})

        ax.stock_img()
        ax.set_extent([-107, -103, 39, 41], crs=ccrs.PlateCarree())
        # ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.7, zorder = 2)
        # ax.add_feature(cfeature.OCEAN, facecolor='black', alpha=0.55, zorder = 2)
        # ax.add_feature(cfeature.COASTLINE, edgecolor='black', zorder = 3)
        # lakes_lambert = lakes_gdf.to_crs(projection.proj4_init)
        # lakes_lambert.plot(ax=ax, facecolor='#73ADFF', edgecolor="#0b4a7a", linewidth=1, alpha=0.7, zorder = 3)

        im = dat_ds_5km_orbit['cosVZA'].plot(
            ax=ax, x='lon', y='lat', transform=ccrs.PlateCarree(),
            alpha=1, cmap='hot',
            vmin=0, vmax=1,
            add_colorbar=False, zorder = 4
        )

        # Add a gray box behind the colorbar
        cbar_box = plt.Rectangle(
            (0.22, 0.28), 0.22, 0.12,  # [left, bottom], width, height (slightly larger than cbar_ax)
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
        cbar.set_label(f'{zlab}', fontsize=23, labelpad=10)
        cbar.ax.tick_params(labelsize=23)
        cbar.set_ticks([0, 1])
        
        ax.set_title('')
        outfile = os.path.join(plot_dir, f'MCD19A1_tile_edge_cosVZA_5km_map_rawnc_{composite_year:04d}-{plot_doy:03d}_orbit_{iorbit:02d}.png')
        plt.savefig(outfile, dpi=100, bbox_inches='tight')
        plt.close()
        print('plot saved to: ', os.path.basename(outfile))


end_period_time = time.time()
elapsed_period = end_period_time - start_period_time
print(f'DOY {plot_doy} plotting took {elapsed_period:.2f} seconds ({td_min(timedelta(seconds=elapsed_period))} minutes and {td_sec(timedelta(seconds=elapsed_period))} seconds).')




#     # %%
# if __name__ == "__main__":
#     main()
# %%
