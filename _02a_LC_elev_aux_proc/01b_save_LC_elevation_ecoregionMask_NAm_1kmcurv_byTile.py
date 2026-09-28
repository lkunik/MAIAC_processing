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

LC_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/ESA_CCI/LandCover/'
forest_frac_file = os.path.join(LC_dir, 'forest_fraction_0.01deg_north_america.nc')
crop_frac_file = os.path.join(LC_dir, 'crop_fraction_0.01deg_north_america.nc')
otherveg_frac_file = os.path.join(LC_dir, 'other_vegetation_fraction_0.01deg_north_america.nc')

elev_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/USGS/GTOPO30/01d/'
elev_file = os.path.join(elev_dir, 'GTOPO30_DEM_01d_NAm.nc')

LC_out_basedir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/ESA_CCI/LandCover/for_MAIAC_proc/'

forest_frac_outdir = os.path.join(LC_out_basedir, 'forest_fraction/')
crop_frac_outdir = os.path.join(LC_out_basedir, 'crop_fraction/')
otherveg_frac_outdir = os.path.join(LC_out_basedir, 'other_vegetation_fraction/')
Path(forest_frac_outdir).mkdir(parents=True, exist_ok=True)
Path(crop_frac_outdir).mkdir(parents=True, exist_ok=True)
Path(otherveg_frac_outdir).mkdir(parents=True, exist_ok=True)

elev_out_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/USGS/GTOPO30/for_MAIAC_proc/'
Path(elev_out_dir).mkdir(parents=True, exist_ok=True)


# EPA_ecoregion_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/EPA/ecoregions/L3/NAm_shp_FOR_MAIAC_PROC/'
# EPA_ecoregion_L3_file = os.path.join(EPA_ecoregion_dir, 'NAm_cec_eco_l3_v2.shp')
# EPA_ecoregion_regrid_out_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/EPA/ecoregions/L3/for_MAIAC_proc/NAm/'
EPA_ecoregion_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/EPA/ecoregions/L2/NAm_shp_FOR_MAIAC_PROC/'
EPA_ecoregion_L2_file = os.path.join(EPA_ecoregion_dir, 'NA_CEC_Eco_Level2_withOther.shp')
EPA_ecoregion_regrid_out_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/EPA/ecoregions/L2/for_MAIAC_proc/NAm/'

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
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': crs}) # establish the figure, axes
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

MODIS_tiles = [	
    # 'h07v06',  # screen 1
    # 'h08v04',
    # 'h08v05',
	# 'h08v06',
	# 'h08v07', 
    # 'h09v02',#<----------------THIS ONE HAS ISSUES REGRIDDING... never completes. Likely because crosses 180 meridian
	# 'h09v03',  
    # 'h09v04',
    # 'h09v05', # screen 2
	# 'h09v06',
	# 'h09v07', 
	'h09v08',  
	# 'h10v02', 
	# 'h10v03',
    # 'h10v04',
	# 'h10v05',
	# 'h10v06', # screen 3
	# 'h10v07',
	'h10v08', 
	# 'h11v02', #
	# 'h11v03', 
	# 'h11v04', 
	# 'h11v05', 
    # 'h11v06', 
    # 'h11v07',   # screen 4
	# 'h12v02',
	# 'h12v03', 
	# 'h12v04',
	# 'h13v02', 
	# 'h13v03', 
	# 'h13v04',
	# 'h14v02',
    # 'h14v03'
    ]
QC_option = 'Cloud' # options are 'CloudAOD', 'Cloud', 'AOD', or 'None'
sample_year = 2018 # this is the specific year-period that every tile has a sample for
sample_period = 14

hist_years = np.arange(2000, 2025)
DOY_composite_starts = np.arange(1, 366, 16)
hist_composites = range(len(DOY_composite_starts))  # 0-22

refl_vars = ['Sur_refl1', 'Sur_refl2', 'Sur_refl4', 'Sur_refl6', 'Sur_refl7', 'Sur_refl11']
#########################################
# Begin main
#########################################
#%%
# def main():




# Load in Land Cover dataset and regrid to 1km curvilinear grid
forest_frac_xr_raw = xr.open_dataset(forest_frac_file)['fraction_forest']
crop_frac_xr_raw = xr.open_dataset(crop_frac_file)['fraction_crop']
otherveg_frac_xr_raw = xr.open_dataset(otherveg_frac_file)['fraction_otherveg']


elev_xr_raw = xr.open_dataset(elev_file)['elevation']

# Set the lowest latitude values to Nan so that any outer edge pixels steal "nan" as the nearest value
forest_frac_xr_raw.loc[dict(y=forest_frac_xr_raw.y.sel(y=0, method='nearest'))] = np.nan
crop_frac_xr_raw.loc[dict(y=crop_frac_xr_raw.y.sel(y=0, method='nearest'))] = np.nan
otherveg_frac_xr_raw.loc[dict(y=otherveg_frac_xr_raw.y.sel(y=0, method='nearest'))] = np.nan
elev_xr_raw.loc[dict(y=elev_xr_raw.y.sel(y=0, method='nearest'))] = np.nan

# Set the lowest longitude values to Nan so that any outer edge pixels steal "nan" as the nearest value
forest_frac_xr_raw.loc[dict(x=forest_frac_xr_raw.x.sel(x=0, method='nearest'))] = np.nan
crop_frac_xr_raw.loc[dict(x=crop_frac_xr_raw.x.sel(x=0, method='nearest'))] = np.nan
otherveg_frac_xr_raw.loc[dict(x=otherveg_frac_xr_raw.x.sel(x=0, method='nearest'))] = np.nan
elev_xr_raw.loc[dict(x=elev_xr_raw.x.sel(x=0, method='nearest'))] = np.nan


# ecoregion_l3_gdf = gpd.read_file(EPA_ecoregion_L3_file)
# NAm_ecoregions = np.unique(ecoregion_l3_gdf.NA_L3NAME)
ecoregion_l2_gdf = gpd.read_file(EPA_ecoregion_L2_file)
NAm_ecoregions = np.unique(ecoregion_l2_gdf.NA_L2NAME)

# set up blank ecoregion masks on the 1km curvilinear grid
xr_ecoregion_mask = xr.zeros_like(elev_xr_raw, dtype=int)
xr_ecoregion_mask = xr_ecoregion_mask.rio.write_crs("EPSG:4326")

#%%
MODIS_tile = MODIS_tiles[0] 
for MODIS_tile in MODIS_tiles:

    print(f'forming regridder for MODIS tile {MODIS_tile}')
    dat_dir = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/pre-fill/QCfilt_{QC_option}/{MODIS_tile}/'
    MCD19A1_files = sorted(glob.glob(os.path.join(dat_dir, f'MCD19A1.{MODIS_tile}.*.nc')))

    sample_file = os.path.join(dat_dir, f'MCD19A1.{MODIS_tile}.061_01d_16day_CV-MVC_QCfilt_{QC_option}_{sample_year}-{sample_period}.nc')
    sample_1km_curv_ds = xr.open_dataset(sample_file).squeeze('time')['NDVI']

    # Get the tile corners from the metadata
    tile_corners = MODIS_tilecorners_df.loc[MODIS_tile]

    regridder_made = False
    forest_frac_out_file = os.path.join(forest_frac_outdir, f'forest_frac_1km_curv_{MODIS_tile}.nc')
    if os.path.exists(forest_frac_out_file):
        print(f'Forest fraction regrid file for {MODIS_tile} already exists, skipping...')
    else:
        forest_frac_xr_tileSubset = forest_frac_xr_raw.sel(x=slice(tile_corners['lon_min']-0.05, tile_corners['lon_max']+0.05),
                                                                y=slice(tile_corners['lat_min']-0.05, tile_corners['lat_max']+0.05))
        
        print('ESA CCI LC regridder...')
        regridder_ESA_CCI = xe.Regridder(forest_frac_xr_tileSubset, sample_1km_curv_ds, "nearest_s2d", unmapped_to_nan = True) # Use XESMF package to establish a regridder which we can recycle
        regridder_made = True
        forest_frac_xr_curv = regridder_ESA_CCI(forest_frac_xr_tileSubset)
        forest_frac_xr_curv.to_netcdf(forest_frac_out_file)
        print('Forest fraction regrid complete.')

        del forest_frac_xr_curv
        gc.collect()

    crop_frac_out_file = os.path.join(crop_frac_outdir, f'crop_frac_1km_curv_{MODIS_tile}.nc')
    if os.path.exists(crop_frac_out_file):
        print(f'Crop fraction regrid file for {MODIS_tile} already exists, skipping...')
    else:
        crop_frac_xr_tileSubset = crop_frac_xr_raw.sel(x=slice(tile_corners['lon_min']-0.05, tile_corners['lon_max']+0.05),
                                                        y=slice(tile_corners['lat_min']-0.05, tile_corners['lat_max']+0.05))
        if not regridder_made:
            print('ESA CCI LC regridder...')
            regridder_ESA_CCI = xe.Regridder(crop_frac_xr_tileSubset, sample_1km_curv_ds, "nearest_s2d", unmapped_to_nan = True) # Use XESMF package to establish a regridder which we can recycle
            regridder_made = True
        crop_frac_xr_curv = regridder_ESA_CCI(crop_frac_xr_tileSubset)
        crop_frac_xr_curv.to_netcdf(crop_frac_out_file)
        print('Crop fraction regrid complete.')

        del crop_frac_xr_curv
        gc.collect()

    otherveg_frac_out_file = os.path.join(otherveg_frac_outdir, f'otherveg_frac_1km_curv_{MODIS_tile}.nc')
    if os.path.exists(otherveg_frac_out_file):
        print(f'Other vegetation fraction regrid file for {MODIS_tile} already exists, skipping...')
    else:
        otherveg_frac_xr_tileSubset = otherveg_frac_xr_raw.sel(x=slice(tile_corners['lon_min']-0.05, tile_corners['lon_max']+0.05),
                                                        y=slice(tile_corners['lat_min']-0.05, tile_corners['lat_max']+0.05))
        if not regridder_made:
            print('ESA CCI LC regridder...')
            regridder_ESA_CCI = xe.Regridder(otherveg_frac_xr_tileSubset, sample_1km_curv_ds, "nearest_s2d", unmapped_to_nan = True) # Use XESMF package to establish a regridder which we can recycle
            regridder_made = True
        otherveg_frac_xr_curv = regridder_ESA_CCI(otherveg_frac_xr_tileSubset)
        otherveg_frac_xr_curv.to_netcdf(otherveg_frac_out_file)
        print('Other vegetation fraction regrid complete.')

        del otherveg_frac_xr_curv
        gc.collect()

    if regridder_made:
        del regridder_ESA_CCI
        gc.collect()


    # # Load in Elevation dataset and regrid to 1km curvilinear grid
    elev_regridder_made = False
    elev_out_file = os.path.join(elev_out_dir, f'GTOPO30_DEM_1km_curv_{MODIS_tile}.nc')
    if os.path.exists(elev_out_file):
        print(f'Elevation regridder for {MODIS_tile} already exists, skipping...')
    else:
        elev_xr_tileSubset = elev_xr_raw.sel(x=slice(tile_corners['lon_min']-0.05, tile_corners['lon_max']+0.05),
                                                        y=slice(tile_corners['lat_min']-0.05, tile_corners['lat_max']+0.05))
        print('Elevation regridder...')
        regridder_elev = xe.Regridder(elev_xr_tileSubset, sample_1km_curv_ds, "nearest_s2d", unmapped_to_nan = True)
        elev_regridder_made = True
        elev_xr_curv = regridder_elev(elev_xr_tileSubset)
        elev_xr_curv.to_netcdf(elev_out_file)
        print('Elevation regridder complete.')

        # test_plot(elev_xr_curv, plot_title='elevation regrid', zlab='elevation')
        del elev_xr_curv
        gc.collect()

    # Loop through the ecoregions and create masks
    print('Looping through ecoregions to create masks...')
    NAm_ecoregion = 'ALASKA BOREAL INTERIOR' # NAm_ecoregions[0] #
    for NAm_ecoregion in NAm_ecoregions:
        ecoregion_mask_outfile = os.path.join(EPA_ecoregion_regrid_out_dir, f'EPA_L2_ecoregion_mask_1km_curv_{MODIS_tile}_{NAm_ecoregion.replace(" ", "_").replace("/", "_")}.nc')

        if os.path.exists(ecoregion_mask_outfile):
            print(f'Ecoregion mask for {NAm_ecoregion} already exists, skipping...')
            continue

        
        region_geom = ecoregion_l2_gdf[ecoregion_l2_gdf.NA_L2NAME == NAm_ecoregion].geometry
        
        xr_ecoregion_mask_tileSubset = xr_ecoregion_mask.sel(x=slice(tile_corners['lon_min']-0.05, tile_corners['lon_max']+0.05),
                                                        y=slice(tile_corners['lat_min']-0.05, tile_corners['lat_max']+0.05))
        xr_ecoregion_mask_clip = xr_ecoregion_mask_tileSubset.astype(float).rio.clip(region_geom, drop=False, invert=False)

        # xr_ecoregion_mask_clipdrop = xr_ecoregion_mask_tileSubset.astype(float).rio.clip(region_geom, drop=True, invert=False)
        # plt.figure(figsize=(10, 8))
        # xr_ecoregion_mask_clipdrop.plot(x='x', y='y', cmap='viridis', add_colorbar=True)
        # plt.title(f'Ecoregion Mask Clip: {NAm_ecoregion}')
        # plt.xlabel('Longitude')
        # plt.ylabel('Latitude')
        # plt.show()

        # Check if the mask is empty after clipping
        if xr_ecoregion_mask_clip.count().item() == 0:
            print(f'Ecoregion mask for {NAm_ecoregion} is empty after clipping, skipping...')
            continue

        print(f'Ecoregion masking for {NAm_ecoregion}')
        # Set all invalid values in xr_ecoregion_mask_clip to -1
        if not elev_regridder_made:
            print('Elevation regridder...')
            regridder_elev = xe.Regridder(xr_ecoregion_mask_tileSubset, sample_1km_curv_ds, "nearest_s2d", unmapped_to_nan = True)
            elev_regridder_made = True
        xr_ecoregion_mask_clip = xr_ecoregion_mask_clip.where(xr_ecoregion_mask_clip > -1, -1)
        xr_ecoregion_mask_clip+=1  # set invalid values to 0, valid values to 1

        xr_ecoregion_mask_curv = regridder_elev(xr_ecoregion_mask_clip)
        xr_ecoregion_mask_curv = xr_ecoregion_mask_curv.where(xr_ecoregion_mask_curv >=0, np.nan)
        if np.nansum(xr_ecoregion_mask_curv == 1) > 0:
            xr_ecoregion_mask_curv = xr_ecoregion_mask_curv.where(xr_ecoregion_mask_curv >= 0, np.nan)
            xr_ecoregion_mask_curv.to_netcdf(ecoregion_mask_outfile)
            print(f'Ecoregion mask for {NAm_ecoregion} saved to {ecoregion_mask_outfile}')
        else:
            print(f'Ecoregion mask for {NAm_ecoregion} regridded but no valid overlapping ecoregion pixels present in dataset, skipping...')
        
    if elev_regridder_made:
        del regridder_elev
        gc.collect()


# # %%

# %%
