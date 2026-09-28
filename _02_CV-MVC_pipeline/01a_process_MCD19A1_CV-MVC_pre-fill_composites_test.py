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


# MCD19A1_dir = "/scratch/general/nfs1/u6017162/MAIAC/pre_proc/raw_nc/"
MCD19A1_dir = "/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/pre_proc/raw_nc/"
out_basedir = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/pre-fill/'
plot_basedir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/plots/maps/MAIAC_CV-MVC/1km_curv/pre-fill/'




#########################################
# Define Global functions
#########################################

def floor(value, decimal_digits):
  scale = (10**decimal_digits)
  return  np.floor(value * scale)/scale

def ceil(value, decimal_digits):
  scale = (10**decimal_digits)
  return  np.ceil(value * scale)/scale


def binary(str_in, num_bits = 16):

    bit_format=f'0{num_bits}b'
    if np.isnan(str_in):
        return num_bits*'1'
    else:
        return format(int(str_in), bit_format)

binary_v = np.vectorize(binary)

def get_bits(str, start_bit, end_bit):
    if start_bit == 0:
        str_split = [*str][-(end_bit+1):]
    else:
        str_split = [*str][-(end_bit+1):-(start_bit)]
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

def clip_2dcurv_ds_to_point(ds, lon_pt, lat_pt):
    # Find the x, y indices where lon_pt and lat_pt are found in ds
    lon_2d = ds.lon.values
    lat_2d = ds.lat.values

    # Compute the absolute difference and find the indices of the minimum
    dist = np.abs(lon_2d - lon_pt) + np.abs(lat_2d - lat_pt)
    y_idx, x_idx = np.unravel_index(np.argmin(dist), lon_2d.shape)
    lon_selected = ds.lon.isel(x=x_idx, y=y_idx).values
    lat_selected = ds.lat.isel(x=x_idx, y=y_idx).values
    print(f"Nearest point in dataset to ({lon_pt}, {lat_pt}) is at ({lon_selected}, {lat_selected}) with distance {dist[y_idx, x_idx]:.6f}")
    ds_point = ds.isel(x=x_idx, y=y_idx)
    return ds_point


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
plot_extent = [-128, -106, 39, 51]  # [minx, maxx, miny, maxy], bounds for western US



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
        plot_min = float(dat_da.nanmin())
    if plot_max is None:
        plot_max = float(dat_da.nanmax())
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
#%%

# Extent of the larger grid, to be subset for this tile
overall_lon_min = -170
overall_lon_max = -50
overall_lat_min = 10
overall_lat_max = 70

# refl_varnames = ['Sur_refl1', 'Sur_refl2', 'Sur_refl4', 'Sur_refl6', 'Sur_refl7', 'Sur_refl11', 'Sigma_BRFn1', 'Sigma_BRFn2']
refl_varnames = ['Sur_refl1', 'Sur_refl2', 'Sur_refl11', 'Sigma_BRFn1', 'Sigma_BRFn2']

# nc_vars1km =["Orbits", "lon", "lat", "Sur_refl1", "Sur_refl2", "Sur_refl4", "Sur_refl6", "Sur_refl7", "Sur_refl11", "Sigma_BRFn1", "Sigma_BRFn2", "Status_QA"]
# nc_vars5km =["Orbits", "lon", "lat", "cosVZA", "cosSZA", "RelAZ", "Scattering_Angle", "Glint_Angle"]

nc_vars1km =["Orbits", "lon", "lat", "Sur_refl1", "Sur_refl2",  "Sur_refl11", "Sigma_BRFn1", "Sigma_BRFn2", "Status_QA"]
nc_vars5km =["Orbits", "lon", "lat", "cosVZA"]


# view zenith angle above which we will exclude obs for "case 2", based on "HIGHVIEW" flag here
# https://landweb.modaps.eosdis.nasa.gov/NPP/forPage/VIIRS_NPP_Surf_Refl_UserGuide_v1.1.pdf
# VZA_thresh = 45 

min_Valid_obs_for_composite = 4

all_MCD19A1_start_dt = dt(2000,1,1)
all_MCD19A2_end_dt = dt(2025, 1, 1)

refl_scale = 0.0001
refl_valid_min = -100 * refl_scale
refl_valid_max = 16000 * refl_scale

cosVZA_scale = 0.0001
cosVZA_valid_min = 0 * cosVZA_scale
cosVZA_valid_max = 10000 * cosVZA_scale

cosSZA_scale = 0.0001
cosSZA_valid_min = 0 * cosSZA_scale
cosSZA_valid_max = 10000 * cosSZA_scale

RelAZ_scale = 0.01
RelAZ_valid_min = -18000 * RelAZ_scale
RelAZ_valid_max = 18000 * RelAZ_scale

Scattering_Angle_scale = 0.01
Scattering_Angle_valid_min = -18000 * Scattering_Angle_scale
Scattering_Angle_valid_max = 18000 * Scattering_Angle_scale

Glint_Angle_scale = 0.01
Glint_Angle_valid_min = -18000 * Glint_Angle_scale
Glint_Angle_valid_max = 18000 * Glint_Angle_scale

# Site coordinates, date ranges, and save paths
site_info = {
    'US-NR1': {
        'tile': 'h09v04',
        'coords': (-105.5464, 40.0329),
        'start': '2017-01-01',
        'end': '2018-12-31'
    },
    'OSBS': {
        'tile': 'h10v06',
        'coords': (-81.993431, 29.689282),
        'start': '2021-01-01',
        'end': '2022-12-31'
    },
    'Ca-Obs': {
        'tile': 'h11v03',
        'coords': (-105.1178, 53.9872),
        'start': '2018-01-01',
        'end': '2022-12-31'
    },
    'DEJU': {
        'tile': 'h11v02',
        'coords': (-145.75136, 63.88112),
        'start': '2019-01-01',
        'end': '2021-12-31'
    }
}
#########################################
# Begin main
#########################################
#%%



# mark start time to keep track of elapsed
start_total = time.time()

MODIS_tile = 'h08v05' #sys.argv[1] #'h10v04'
composite_year = 2001 # command line argument
composite_period = 20 # command line argument
    
# MODIS_tile = 'h10v06'
# composite_year = 2021 # command line argument
# composite_period = 17 # command line argument


# CloudFree options (6):
# 'CloudFree_LowAOD_ClearAdj'      # Strictest cloud, low AOD, strictest adjacency
# 'CloudFree_LowAOD_NotSurround'   # Strictest cloud, low AOD, moderate adjacency
# 'CloudFree_LowAOD_AllAdj'        # Strictest cloud, low AOD, no adjacency filter
# 'CloudFree_AllAOD_ClearAdj'       # Strictest cloud, no AOD filter, strictest adjacency
# 'CloudFree_AllAOD_NotSurround'    # Strictest cloud, no AOD filter, moderate adjacency
# 'CloudFree_AllAOD_AllAdj'         # Strictest cloud only, no AOD or adjacency filters

# LooseCloud options (6):
# 'LooseCloud_LowAOD_ClearAdj'     # Moderate cloud, low AOD, strictest adjacency
# 'LooseCloud_LowAOD_NotSurround'  # Moderate cloud, low AOD, moderate adjacency
# 'LooseCloud_LowAOD_AllAdj'       # Moderate cloud, low AOD, no adjacency filter
# 'LooseCloud_AllAOD_ClearAdj'      # Moderate cloud, no AOD filter, strictest adjacency
# 'LooseCloud_AllAOD_NotSurround'   # Moderate cloud, no AOD filter, moderate adjacency
# 'LooseCloud_AllAOD_AllAdj'        # Moderate cloud only, no AOD or adjacency filters

# AllCloud options (6):
# 'AllCloud_LowAOD_ClearAdj'       # No cloud filter, low AOD, strictest adjacency
# 'AllCloud_LowAOD_NotSurround'    # No cloud filter, low AOD, moderate adjacency
# 'AllCloud_LowAOD_AllAdj'         # No cloud filter, low AOD only
# 'AllCloud_AllAOD_ClearAdj'        # No cloud or AOD filters, strictest adjacency only
# 'AllCloud_AllAOD_NotSurround'     # No cloud or AOD filters, moderate adjacency only
# 'AllCloud_AllAOD_AllAdj'          # No filtering applied (equivalent to raw data)
QC_option = 'CloudFree_LowAOD_ClearAdj' # command line argument - options are 'CloudFree', 'CloudAOD', 'Cloud', 'AOD', or 'None'

VZA_min = 0
VZA_max = 90

if VZA_min > 0 or VZA_max < 90:
    out_dir = os.path.join(out_basedir, f'VZA{VZA_min}-{VZA_max}/QCfilt_{QC_option}', MODIS_tile)
else:
    out_dir = os.path.join(out_basedir, f'QCfilt_{QC_option}', MODIS_tile)

MCD19A1_files = sorted(glob.glob(f'{MCD19A1_dir}MCD19A1*{MODIS_tile}*.nc')) #
Path(out_dir).mkdir(parents=True, exist_ok=True)
outfile = os.path.join(out_dir, f'MCD19A1.{MODIS_tile}.061_1km_16day_CV-MVC_QCfilt_{QC_option}_{composite_year}-{str(composite_period).zfill(2)}.nc')

plot_dir = os.path.join(plot_basedir, f'QCfilt_{QC_option}', MODIS_tile)
Path(plot_dir).mkdir(parents=True, exist_ok=True)

MCD19A1_filedates = [dt.strptime(os.path.basename(file)[9:16], '%Y%j').date() for file in MCD19A1_files]
MCD19A1_filedoys = [int(date.strftime('%j')) for date in MCD19A1_filedates]
DOY_composite_starts = np.arange(1, 366, 16)
MCD19A1_file_composite_periods = np.searchsorted(DOY_composite_starts, MCD19A1_filedoys, side='right') - 1


if os.path.exists(outfile):
    print(f'Output file {os.path.basename(outfile)} already exists, continuing anyways for testing purposes')
    # print(f'Output file {os.path.basename(outfile)} already exists, skipping processing for {composite_year}-{str(composite_period).zfill(2)}')
    # sys.exit(0)


iYYMM = [ i for i in range(len(MCD19A1_filedates)) if ((MCD19A1_filedates[i].year == composite_year) &
                                                (MCD19A1_file_composite_periods[i] == composite_period)) ]
dat_xrs = []

if len(iYYMM) == 0:
    print(f'No files found for {composite_year}-{str(composite_period).zfill(2)}, skipping processing')
    # sys.exit(0)

tot_gridcells_16day = 0
tot_orig_nonNA_16day = 0

### Produce XESMF regridder objects for 1km and 5km data before looping through files
isample = 0
sample_nc_file = MCD19A1_files[iYYMM[isample]]
while os.stat(sample_nc_file).st_size == 0:
    isample += 1
    if isample >= len(iYYMM):
        print(f'No non-empty files found for {composite_year}-{str(composite_period).zfill(2)}, cannot create regridder, exiting')
        # sys.exit(0)
    sample_nc_file = MCD19A1_files[iYYMM[isample]]

try:
    # open a sample file to get lat/lon grid info
    MODIS1km_xr_sample = xr.open_dataset(sample_nc_file, group = "grid1km/Data_Fields")[nc_vars1km]
    MODIS1km_xr_sample = MODIS1km_xr_sample.rename({'XDim': 'x', 'YDim': 'y'})
    MODIS1km_xr_sample.rio.write_crs(4326, inplace = True) # specify WGS84 as CRS
    MODIS5km_xr_sample = xr.open_dataset(sample_nc_file, group = "grid5km/Data_Fields")[nc_vars5km]
    MODIS5km_xr_sample = MODIS5km_xr_sample.rename({'XDim': 'x', 'YDim': 'y'})
    MODIS5km_xr_sample.rio.write_crs(4326, inplace = True) # specify WGS84 as CRS
except OSError as e:
    print(f'ERROR opening sample file {os.path.basename(sample_nc_file)}: {e}')
    print(f'Cannot create regridder without sample file, exiting')
    # sys.exit(1)

regridder_5km_to_1km = xe.Regridder(MODIS5km_xr_sample, MODIS1km_xr_sample, 'bilinear', extrap_method = 'nearest_s2d') # extrap method is important for edge case near tile boundary!

# num_iter = 8
# ii = iYYMM[num_iter] 
for num_iter, ii in enumerate(iYYMM):
    
    file = MCD19A1_files[ii]
    filedate = MCD19A1_filedates[ii]
    if os.stat(file).st_size == 0:
        print(f'    File {os.path.basename(file)} is empty, skipping')
        continue
    # print(f'    loading file: {os.path.basename(file)} - 1km data')

    ### Open netcdf file using xarray package:
    # 1km data
    try:
        MODIS1km_xr = xr.open_dataset(file, group = "grid1km/Data_Fields")[nc_vars1km]
    
        MODIS1km_xr = MODIS1km_xr.rename({'XDim': 'x', 'YDim': 'y'})
        xr1km_Orbits = MODIS1km_xr['Orbits']
        xr1km_y = MODIS1km_xr['y']
        xr1km_x = MODIS1km_xr['x']
        MODIS1km_xr.rio.write_crs(4326, inplace = True) # specify WGS84 as CRS


        with xr.open_dataset(file) as ds:
            # MODIS1km_xr_orbit_amounts = ds.attrs.get('Orbit_amount_GLOSDS', None)
            MODIS1km_xr_orbit_time_stamp_strings = ds.attrs.get('Orbit_time_stamp_GLOSDS', None)

        # print_timestamp_satellite_info_from_global_attrs(MODIS1km_xr_orbit_time_stamp_strings)
        MODIS1km_xr_orbit_time_stamps = parse_timestamps_to_datetime_from_global_attrs(MODIS1km_xr_orbit_time_stamp_strings)

        # Define relevant QA bits
        # see pg 12 of https://lpdaac.usgs.gov/documents/1500/MCD19_User_Guide_V61.pdf
        MODIS1km_xr_QA_bits = binary_v(MODIS1km_xr['Status_QA'])
        MODIS1km_xr_cloud_mask = get_bits_v(MODIS1km_xr_QA_bits, 0, 2)
        MODIS1km_xr_LWSI_mask = get_bits_v(MODIS1km_xr_QA_bits, 3, 4)
        MODIS1km_xr_adjacency_mask = get_bits_v(MODIS1km_xr_QA_bits, 5, 7)
        MODIS1km_xr_AOD_lvl = get_bits_v(MODIS1km_xr_QA_bits, 8, 8)
        # MODIS1km_xr_AOD_type = get_bits_v(MODIS1km_xr_QA_bits, 9, 10)
        # MODIS1km_xr_BRF_over_snow = get_bits_v(MODIS1km_xr_QA_bits, 11, 11)
        # MODIS1km_xr_high_altitude = get_bits_v(MODIS1km_xr_QA_bits, 12, 12)
        # MODIS1km_xr_surf_change_mask = get_bits_v(MODIS1km_xr_QA_bits, 13, 15)

        # filter for valid values, see pg 11 of https://lpdaac.usgs.gov/documents/1500/MCD19_User_Guide_V61.pdf
        # But note that scale is already applied by the hdf-to-nc converter
        # loop through all bands downloaded
        for refl_var in refl_varnames:
            MODIS1km_xr = MODIS1km_xr.where((MODIS1km_xr[refl_var] >= refl_valid_min) & (MODIS1km_xr[refl_var] < refl_valid_max))

        # Total pre-QC obs across all pixels for this file
        num_total_preQC = MODIS1km_xr.Sur_refl1.count().values
        print(f'Band 1: # of non-nan values before QC = {num_total_preQC}')

        nan_mask = np.where(MODIS1km_xr_QA_bits == 16*'1', 1, 0) # 1 if all bits are 1 (i.e. invalid), 0 otherwise

        # Apply all Cloud QC filtering options
        cloud_mask_clearPossiblyCloudy = np.where(np.isin(MODIS1km_xr_cloud_mask, ['001', '010']), 1, 0) # 1 if "clear" or "possibly cloudy", 0 otherwise
        cloud_mask_ClearOnly = np.where(MODIS1km_xr_cloud_mask == '001', 1, 0) # 1 if "clear", 0 otherwise

        # LWSI_mask_land = np.where(MODIS1km_xr_LWSI_mask == '00', 1, 0) # 1 if "Land", 0 otherwise
        # LWSI_mask_water = np.where(MODIS1km_xr_LWSI_mask == '01', 1, 0) # 1 if "Water", 0 otherwise
        LWSI_mask_snowIce = np.where(np.isin(MODIS1km_xr_LWSI_mask, ['10', '11']), 1, 0) # 1 if "snow" or "ice", 0 otherwise

        # Apply all AOD QC filtering options
        AOD_mask_LowOnly = np.where(MODIS1km_xr_AOD_lvl == '0', 1, 0) # only want low AOD

        # Apply all Adjacency Mask QC filtering options
        adjacency_mask_ClearOnly = np.where(MODIS1km_xr_adjacency_mask == '000', 1, 0) # Flag if not "Normal condition/Clear"
        adjacency_mask_NotCloudSurround = np.where(MODIS1km_xr_adjacency_mask == '010', 0, 1) # Flag if "Surrounded by more than 4 cloudy pixels"
        


        # # don't care about remainder of QA bits for now...

        # BRF_over_snow_mask = np.where(MODIS1km_xr_BRF_over_snow == '1', 1, 0) # Flag if "BRF over snow"

        # del MODIS1km_xr_QA_bits, MODIS1km_xr_cloud_mask, MODIS1km_xr_AOD_lvl # dump large unneeded vars from env
        # gc.collect() # manual garbage collection

        # add QA masks to the 1km data
        MODIS1km_xr = MODIS1km_xr.drop_vars('Status_QA')

        # Define cloud mask options:
        # "CloudFree" = only include obs flagged as "clear" in the QA (i.e. exclude "possibly cloudy")
        # "LooseCloud" = include obs flagged as "clear" or "possibly cloudy" in the QA (i.e. exclude only "cloudy")
        # "AllCloud" = absence of any filtering (all values allowed)
        MODIS1km_xr = MODIS1km_xr.assign(all_nan_mask = xr.DataArray(nan_mask, coords={'Orbits':xr1km_Orbits, 'y':xr1km_y, 'x':xr1km_x}, dims=['Orbits', 'y', 'x']))
        MODIS1km_xr = MODIS1km_xr.assign(LooseCloud = xr.DataArray(cloud_mask_clearPossiblyCloudy, coords={'Orbits':xr1km_Orbits, 'y':xr1km_y, 'x':xr1km_x}, dims=['Orbits', 'y', 'x']))
        MODIS1km_xr = MODIS1km_xr.assign(CloudFree = xr.DataArray(cloud_mask_ClearOnly, coords={'Orbits':xr1km_Orbits, 'y':xr1km_y, 'x':xr1km_x}, dims=['Orbits', 'y', 'x']))

        MODIS1km_xr = MODIS1km_xr.assign(SnowIce = xr.DataArray(LWSI_mask_snowIce, coords={'Orbits':xr1km_Orbits, 'y':xr1km_y, 'x':xr1km_x}, dims=['Orbits', 'y', 'x']))
        # MODIS1km_xr = MODIS1km_xr.assign(Land = xr.DataArray(LWSI_mask_land, coords={'Orbits':xr1km_Orbits, 'y':xr1km_y, 'x':xr1km_x}, dims=['Orbits', 'y', 'x']))
        # MODIS1km_xr = MODIS1km_xr.assign(Water = xr.DataArray(LWSI_mask_water, coords={'Orbits':xr1km_Orbits, 'y':xr1km_y, 'x':xr1km_x}, dims=['Orbits', 'y', 'x']))
        # Define AOD mask options:
        # "LowAOD" = only include obs flagged as "low AOD" in the QA (i.e. exclude "high AOD")
        # "AllAOD" = absence of any filtering (all values allowed)
        MODIS1km_xr = MODIS1km_xr.assign(LowAOD = xr.DataArray(AOD_mask_LowOnly, coords={'Orbits':xr1km_Orbits, 'y':xr1km_y, 'x':xr1km_x}, dims=['Orbits', 'y', 'x']))
        
        # Define Adjacency mask options:
        # "ClearAdj" = only include obs as "Normal condition/Clear" in the QA (i.e. exclude any flagged as "Surrounded by more than 4 cloudy pixels")
        # "NotSurround" = only include obs flagged as "not adjacent to cloudy pixels" in the QA (i.e. exclude obs that are "surrounded by more than 4 cloudy pixels")
        # "AllAdj" = absence of any filtering (all values allowed)
        MODIS1km_xr = MODIS1km_xr.assign(ClearAdj = xr.DataArray(adjacency_mask_ClearOnly, coords={'Orbits':xr1km_Orbits, 'y':xr1km_y, 'x':xr1km_x}, dims=['Orbits', 'y', 'x']))
        MODIS1km_xr = MODIS1km_xr.assign(NotSurround = xr.DataArray(adjacency_mask_NotCloudSurround, coords={'Orbits':xr1km_Orbits, 'y':xr1km_y, 'x':xr1km_x}, dims=['Orbits', 'y', 'x']))
        # MODIS1km_xr = MODIS1km_xr.assign(BRF_over_snow = xr.DataArray(BRF_over_snow_mask, coords={'Orbits':xr1km_Orbits, 'y':xr1km_y, 'x':xr1km_x}, dims=['Orbits', 'y', 'x']))

        MODIS1km_xr = MODIS1km_xr.assign(QC_bits = xr.DataArray(MODIS1km_xr_QA_bits, coords={'Orbits':xr1km_Orbits, 'y':xr1km_y, 'x':xr1km_x}, dims=['Orbits', 'y', 'x']))
        # del cloud_mask_clearPossiblyCloudy, AOD_mask_LowOnly # dump large unneeded vars from env
        # gc.collect() # manual garbage collection

        # drop any flag values where band 1 is null to avoid interpretation of '1111111111111111' 
        # which is the 16-bit unsigned value assigned to NaN values
        MODIS1km_xr = MODIS1km_xr.where(MODIS1km_xr.all_nan_mask == 0) 

        for site in site_info.keys():
            if site_info[site]['tile'] == MODIS_tile:
                print(f"Processing site {site} at coordinates {site_info[site]['coords']} for file {os.path.basename(file)}")
                MODIS1km_xr_PS_site = clip_2dcurv_ds_to_point(MODIS1km_xr, site_info[site]['coords'][0], site_info[site]['coords'][1])
                print(f'    {site}: Cloud mask - CloudFree: {MODIS1km_xr_PS_site.Sur_refl1.where(MODIS1km_xr_PS_site.CloudFree == 1).count().values} valid values')
                print(f'    {site}: Cloud mask - LooseCloud: {MODIS1km_xr_PS_site.Sur_refl1.where(MODIS1km_xr_PS_site.LooseCloud == 1).count().values} valid values')
                print(f'    {site}: Adjacency mask - NotSurround: {MODIS1km_xr_PS_site.Sur_refl1.where(MODIS1km_xr_PS_site.NotSurround == 1).count().values} valid values')
                print(f'    {site}: Adjacency mask - ClearAdj: {MODIS1km_xr_PS_site.Sur_refl1.where(MODIS1km_xr_PS_site.ClearAdj == 1).count().values} valid values')
                print(f'    {site}: AOD mask - LowAOD: {MODIS1km_xr_PS_site.Sur_refl1.where(MODIS1km_xr_PS_site.LowAOD == 1).count().values} valid values')
                print(f'    {site}: Snow/Ice mask - SnowIce: {MODIS1km_xr_PS_site.Sur_refl1.where(MODIS1km_xr_PS_site.SnowIce == 1).count().values} values flagged as Snow/Ice')
                # print(f'    {site}: Land mask - Land: {MODIS1km_xr_PS_site.Sur_refl1.where(MODIS1km_xr_PS_site.Land == 1).count().values} values flagged as Land')
                # print(f'    {site}: Water mask - Water: {MODIS1km_xr_PS_site.Sur_refl1.where(MODIS1km_xr_PS_site.Water == 1).count().values} values flagged as Water')
                # print(f'    {site}: BRF over snow mask - BRF_over_snow: {MODIS1km_xr_PS_site.Sur_refl1.where(MODIS1km_xr_PS_site.BRF_over_snow == 1).count().values} values flagged as BRF over snow')
                print(f'    {site}: QC bits: {MODIS1km_xr_PS_site.QC_bits.values}')
        print(f'Cloud mask - CloudFree: {MODIS1km_xr.Sur_refl1.where(MODIS1km_xr.CloudFree == 1).count().values} valid values')
        print(f'Cloud mask - LooseCloud: {MODIS1km_xr.Sur_refl1.where(MODIS1km_xr.LooseCloud == 1).count().values} valid values')
        print(f'Snow/Ice mask - SnowIce: {MODIS1km_xr.Sur_refl1.where(MODIS1km_xr.SnowIce == 1).count().values} values flagged as Snow/Ice')
        # print(f'Land mask - Land: {MODIS1km_xr.Sur_refl1.where(MODIS1km_xr.Land == 1).count().values} values flagged as Land')
        # print(f'Water mask - Water: {MODIS1km_xr.Sur_refl1.where(MODIS1km_xr.Water == 1).count().values} values flagged as Water')
        print(f'Adjacency mask - NotSurround: {MODIS1km_xr.Sur_refl1.where(MODIS1km_xr.NotSurround == 1).count().values} valid values')
        print(f'Adjacency mask - ClearAdj: {MODIS1km_xr.Sur_refl1.where(MODIS1km_xr.ClearAdj == 1).count().values} valid values')
        print(f'AOD mask - LowAOD: {MODIS1km_xr.Sur_refl1.where(MODIS1km_xr.LowAOD == 1).count().values} valid values')
        # print(f'BRF over snow mask - BRF_over_snow: {MODIS1km_xr.Sur_refl1.where(MODIS1km_xr.BRF_over_snow == 1).count().values} values flagged as BRF over snow')

        ALL_MASKS_TRUE_xr = MODIS1km_xr.Sur_refl1.where((MODIS1km_xr.CloudFree == 1) & 
                                                                (MODIS1km_xr.LowAOD == 1) &
                                                                (MODIS1km_xr.ClearAdj == 1))
        print(f'Strictest Cloud + AOD + Adjacency mask: {ALL_MASKS_TRUE_xr.count().values} valid values')

        # test_plot(MODIS1km_xr.Land.sel(Orbits=0), f'{str(filedate)} Land Mask',
        #               'Land Mask', None, plot_min = 0, plot_max = 1)

        # test_plot(MODIS1km_xr.Water.sel(Orbits=0), f'{str(filedate)} Water Mask',
        #               'Water Mask', None, plot_min = 0, plot_max = 1)

        test_plot(MODIS1km_xr.SnowIce.sel(Orbits=0), f'{str(filedate)} Snow/Ice Mask',
                      'Snow/Ice Mask', None, plot_min = 0, plot_max = 1)

        # test_plot(MODIS1km_xr.BRF_over_snow.sel(Orbits=0), f'{str(filedate)} BRF over snow Mask',
        #               'BRF over snow Mask', None, plot_min = 0, plot_max = 1)

        # add to the count of total number of possible gridcells*overpasses
        nOrbits = len(MODIS1km_xr['Orbits'].values)
        ny = len(MODIS1km_xr['lat'].values)
        nx = len(MODIS1km_xr['lon'].values)
        tot_gridcells = nOrbits * ny * nx
        tot_gridcells_16day+=tot_gridcells

        # add to the count of total number of non-NAN values after aggregating the pre-QC data
        tot_nonNAN_Band1_preQC = MODIS1km_xr.Sur_refl1.count().values
        tot_nonNAN_Band11_preQC = MODIS1km_xr.Sur_refl11.count().values
        if tot_nonNAN_Band1_preQC != tot_nonNAN_Band11_preQC:
            print(f'Diff number of non-NAN vals for Band 1 ({tot_nonNAN_Band1_preQC}) and Band 11 ({tot_nonNAN_Band11_preQC})')
        tot_orig_nonNA_16day += tot_nonNAN_Band1_preQC


        # Apply the final QC filter based on function input option
        if QC_option == 'CloudFree_LowAOD_ClearAdj':
            MODIS1km_xr_filt = MODIS1km_xr.where((MODIS1km_xr.CloudFree == 1) & (MODIS1km_xr.LowAOD == 1) & (MODIS1km_xr.ClearAdj == 1))
        elif QC_option == 'CloudFree_LowAOD_NotSurround':
            MODIS1km_xr_filt = MODIS1km_xr.where((MODIS1km_xr.CloudFree == 1) & (MODIS1km_xr.LowAOD == 1) & (MODIS1km_xr.NotSurround == 1))
        elif QC_option == 'CloudFree_LowAOD_AllAdj':
            MODIS1km_xr_filt = MODIS1km_xr.where((MODIS1km_xr.CloudFree == 1) & (MODIS1km_xr.LowAOD == 1))
        elif QC_option == 'CloudFree_AllAOD_ClearAdj':
            MODIS1km_xr_filt = MODIS1km_xr.where((MODIS1km_xr.CloudFree == 1) & (MODIS1km_xr.ClearAdj == 1))
        elif QC_option == 'CloudFree_AllAOD_NotSurround':
            MODIS1km_xr_filt = MODIS1km_xr.where((MODIS1km_xr.CloudFree == 1) & (MODIS1km_xr.NotSurround == 1))
        elif QC_option == 'CloudFree_AllAOD_AllAdj':
            MODIS1km_xr_filt = MODIS1km_xr.where(MODIS1km_xr.CloudFree == 1)
        elif QC_option == 'LooseCloud_LowAOD_ClearAdj':
            MODIS1km_xr_filt = MODIS1km_xr.where((MODIS1km_xr.LooseCloud == 1) & (MODIS1km_xr.LowAOD == 1) & (MODIS1km_xr.ClearAdj == 1))
        elif QC_option == 'LooseCloud_LowAOD_NotSurround':
            MODIS1km_xr_filt = MODIS1km_xr.where((MODIS1km_xr.LooseCloud == 1) & (MODIS1km_xr.LowAOD == 1) & (MODIS1km_xr.NotSurround == 1))
        elif QC_option == 'LooseCloud_LowAOD_AllAdj':
            MODIS1km_xr_filt = MODIS1km_xr.where((MODIS1km_xr.LooseCloud == 1) & (MODIS1km_xr.LowAOD == 1))
        elif QC_option == 'LooseCloud_AllAOD_ClearAdj':
            MODIS1km_xr_filt = MODIS1km_xr.where((MODIS1km_xr.LooseCloud == 1) & (MODIS1km_xr.ClearAdj == 1))
        elif QC_option == 'LooseCloud_AllAOD_NotSurround':
            MODIS1km_xr_filt = MODIS1km_xr.where((MODIS1km_xr.LooseCloud == 1) & (MODIS1km_xr.NotSurround == 1))
        elif QC_option == 'LooseCloud_AllAOD_AllAdj':
            MODIS1km_xr_filt = MODIS1km_xr.where(MODIS1km_xr.LooseCloud == 1)
        elif QC_option == 'AllCloud_LowAOD_ClearAdj':
            MODIS1km_xr_filt = MODIS1km_xr.where((MODIS1km_xr.LowAOD == 1) & (MODIS1km_xr.ClearAdj == 1))
        elif QC_option == 'AllCloud_LowAOD_NotSurround':
            MODIS1km_xr_filt = MODIS1km_xr.where((MODIS1km_xr.LowAOD == 1) & (MODIS1km_xr.NotSurround == 1))
        elif QC_option == 'AllCloud_LowAOD_AllAdj':
            MODIS1km_xr_filt = MODIS1km_xr.where((MODIS1km_xr.LowAOD == 1))
        elif QC_option == 'AllCloud_AllAOD_ClearAdj':
            MODIS1km_xr_filt = MODIS1km_xr.where((MODIS1km_xr.ClearAdj == 1))
        elif QC_option == 'AllCloud_AllAOD_NotSurround':
            MODIS1km_xr_filt = MODIS1km_xr.where((MODIS1km_xr.NotSurround == 1))
        elif QC_option == 'AllCloud_AllAOD_AllAdj':
            MODIS1km_xr_filt = MODIS1km_xr
        elif QC_option == 'CloudFree': # REMOVE THIS AFTER FILLING h09v02 FOR THE WILKES POSTER
            MODIS1km_xr_filt = MODIS1km_xr.where(MODIS1km_xr.CloudFree == 1)
        else:
            raise ValueError(f'Invalid QC_option: {QC_option}')


        # plot output for testing:
        MODIS1km_xr_filt_test = MODIS1km_xr_filt.assign(NDVI = (MODIS1km_xr_filt["Sur_refl2"] - MODIS1km_xr_filt["Sur_refl1"])/
                                   (MODIS1km_xr_filt["Sur_refl2"] + MODIS1km_xr_filt["Sur_refl1"]))
        # for Orbit in MODIS1km_xr_filt_test.Orbits.values:
        #    test_plot(MODIS1km_xr_filt_test.Sur_refl1.sel(Orbits=Orbit), 
        #              f'{str(filedate)} Orbit {Orbit}', 'Sur_refl1', None,
        #             #  f'test_NDVI/{Path(file).stem}_1km_NDVI_Orbit{Orbit}.png',
        #              plot_min = 0, plot_max = 0.3)

        # Total pre-QC obs across all pixels for this file
        num_total_postQC = MODIS1km_xr_filt.Sur_refl1.count().values
        print(f'Band 1: # of non-nan values after QC = {num_total_postQC}')

        # print(f'    loading file: {os.path.basename(file)} - 5km data')

        # 5km view angle data
        MODIS5km_xr = xr.open_dataset(file, group = "grid5km/Data_Fields")[nc_vars5km]
        MODIS5km_xr = MODIS5km_xr.rename({'XDim': 'x', 'YDim': 'y'})

        MODIS5km_xr = MODIS5km_xr.where((MODIS5km_xr['cosVZA'] >= cosVZA_valid_min) & (MODIS5km_xr['cosVZA'] < cosVZA_valid_max))

        # print("        restructuring 5km data from multidimensional coords to single dimension coords")

        # for Orbit in MODIS5km_xr.Orbits.values:
        #    test_plot(MODIS5km_xr.cosVZA.sel(Orbits=Orbit), 
        #              f'{str(filedate)} Orbit {Orbit}', 'cosVZA', None,
        #             #  f'test_NDVI/{Path(file).stem}_1km_NDVI_Orbit{Orbit}.png',
        #              plot_min = 0, plot_max = 1)
           

        VZA_arr_5km = rad_to_deg(arccos_v(MODIS5km_xr['cosVZA']))
        MODIS5km_xr = MODIS5km_xr.assign(VZA = xr.DataArray(VZA_arr_5km, coords={'Orbits':MODIS5km_xr.coords['Orbits'].values, 'y':MODIS5km_xr.coords['y'].values, 'x':MODIS5km_xr.coords['x'].values}, dims=['Orbits', 'y', 'x']))
        MODIS5km_xr.rio.write_crs(4326, inplace = True) # specify WGS84 as CRS

        # now resample MODIS5km to match the MODIS1km grid
        MODIS5km_xr_interp = regridder_5km_to_1km(MODIS5km_xr)
        MODIS5km_xr_interp = MODIS5km_xr_interp.where((MODIS5km_xr_interp['cosVZA'] >= cosVZA_valid_min) & (MODIS5km_xr_interp['cosVZA'] < cosVZA_valid_max))

        # check that the number of orbits matches in each grid dataset
        norbits1km = len(MODIS1km_xr.coords['Orbits'])
        norbits5km = len(MODIS5km_xr_interp.coords['Orbits'])
        if ~ norbits1km == norbits5km:
            raise Exception(f"number of orbits doesn't match for file {file}")

        # # convert cosVZA to VZA in degrees
        VZA_arr = rad_to_deg(arccos_v(MODIS5km_xr_interp['cosVZA']))
        MODIS5km_xr_interp = MODIS5km_xr_interp.assign(VZA = xr.DataArray(VZA_arr, coords={'Orbits':MODIS5km_xr_interp.coords['Orbits'].values, 'y':MODIS5km_xr_interp.coords['y'].values, 'x':MODIS5km_xr_interp.coords['x'].values}, dims=['Orbits', 'y', 'x']))
        
        # print("        combining 1km and 5km data, appending to stack")
        # merge the datasets
        MODIS_1km5km_combined = xr.merge([MODIS1km_xr_filt, MODIS5km_xr_interp])
        MODIS_1km5km_combined = MODIS_1km5km_combined.rename({'Orbits': 'time'})
        MODIS_1km5km_combined = MODIS_1km5km_combined.assign_coords(time=MODIS1km_xr_orbit_time_stamps)
        MODIS_1km5km_combined = MODIS_1km5km_combined.where(MODIS_1km5km_combined['Sur_refl1'].notnull())


           
        dat_xrs.append(MODIS_1km5km_combined)

        MODIS1km_xr.close() # close the file connection
        MODIS5km_xr.close()

        # # remove large, unused vars
        # del MODIS1km_xr, MODIS5km_xr, MODIS_1km5km_combined, MODIS5km_xr_interp 
        # del VZA_arr, VZA_arr_5km, MODIS1km_xr_filt 
        # del ALL_MASKS_TRUE_xr
        # gc.collect() # manual garbage collection

        # end_file = time.time()
        # file_elapsed = timedelta(seconds=end_file-start_file)
        # total_elapsed = timedelta(seconds=end_file-start_total)
        # time_remaining = total_elapsed * (len(iYYMM)-num_iter-1)/(num_iter+1)
        # print(f'file processing took {td_min(file_elapsed)} mins, {td_sec(file_elapsed)} seconds')
        # print(f'  estimated time remaining: {td_min(time_remaining)} mins, {td_sec(time_remaining)} seconds')

    except (OSError, RuntimeError, ValueError) as e:
        print(f'    ERROR processing file {os.path.basename(file)}: {e}')
        print(f'    Skipping bad file')
        # Close any open file handles
        try:
            MODIS1km_xr.close()
        except:
            pass
        try:
            MODIS5km_xr.close()
        except:
            pass
        continue

print(f'Done stacking daily files for {composite_year}-{str(composite_period).zfill(2)}. Concatenating into single dataset...')
dat_full_ds = xr.concat(dat_xrs, dim = 'time') # concatenate along time axis
# del dat_xrs # remove large, unused vars
# gc.collect() # manual garbage collection

#####################################
### Implement CV-MVC algorithm
#####################################

#%%
# Calculate NDVI for the CV-MVC algorithm
dat_full_ds = dat_full_ds.assign(NDVI=(dat_full_ds["Sur_refl2"] - dat_full_ds["Sur_refl1"])/
                        (dat_full_ds["Sur_refl2"] + dat_full_ds["Sur_refl1"]))

### Step 0: filter for valid VZA range
if VZA_min > 0 or VZA_max < 90:
    dat_full_ds = dat_full_ds.where((dat_full_ds['VZA'] >= VZA_min) & (dat_full_ds['VZA'] <= VZA_max))

# Get number of valid observations per x, y pixel
dat_full_ds = dat_full_ds.assign(valid_obs_count=dat_full_ds.NDVI.notnull().sum(dim='time'))

### Step 1: pixel-wise sort by NDVI
# define the array from which we want to sort NDVI values
arr_sort_NDVI = -dat_full_ds.NDVI # NOTE THE NEGATIVE!!! makes it a descending sort
sort_axis = arr_sort_NDVI.get_axis_num('time') # this gets the axis over which we want to sort
sort_inds_NDVI = arr_sort_NDVI.argsort(axis=sort_axis) # perform the sort to get pixel-wise indices of the highest NDVI, 2nd highest NDVI, etc...
sort_inds_NDVI = sort_inds_NDVI.rename({'time': 'NDVI_rank'}) # Replace time dim with ranking dim
sort_inds_top2NDVI = sort_inds_NDVI.isel(NDVI_rank = [0,1]) # select only the top 2 NDVI indices
dat_full_ds_top2NDVI = dat_full_ds.isel({'time': sort_inds_top2NDVI}) # filter the original stacked dataset for these top 2 NDVI indices

# del dat_full_ds, sort_inds_NDVI, arr_sort_NDVI
# gc.collect() # manual garbage collection

### Step 2: pixel-wise sort by View angle
# pipe the NDVI-filtered dataset into this step to sort by VZA values
arr_sort_VZA = dat_full_ds_top2NDVI.VZA # NOTE There is NO negative, makes it an ascending sort
sort_inds_VZA = arr_sort_VZA.argsort(axis=sort_axis) # perform sort to get pixel-wise indices of lowest and 2nd lowest VZA (NOTE sort_axis should be the same axis as with NDVI.)
sort_inds_VZA = sort_inds_VZA.rename({'NDVI_rank': 'VZA_rank'}) # replace NDVI_rank dim with VZA_rank
sort_inds_lowVZA = sort_inds_VZA.isel(VZA_rank = 0) # get the index of the lowest VZA
sort_inds_highVZA = sort_inds_VZA.isel(VZA_rank = 1) # get the other index
dat_full_ds_top2NDVI_lowestVZA = dat_full_ds_top2NDVI.isel({'NDVI_rank': sort_inds_lowVZA}) # Using the NDVI-filtered stacked dataset, filter for the lowest VZA
dat_full_ds_top2NDVI_highestVZA = dat_full_ds_top2NDVI.isel({'NDVI_rank': sort_inds_highVZA}) # For any pixels 

# del arr_sort_VZA, sort_inds_VZA, sort_inds_lowVZA, sort_inds_highVZA, dat_full_ds_top2NDVI
# gc.collect() # manual garbage collection

### Step 3: pixel-wise check for <2 valid NDVI vals
# if the resulting NDVI value is nan, use the values of the next smallest VZA.
# (If there was only 1 valid NDVI value to begin, this will recover that value)
nan_inds_NDVI = np.isnan(dat_full_ds_top2NDVI_lowestVZA.NDVI)
composite_ds = dat_full_ds_top2NDVI_lowestVZA.where(~nan_inds_NDVI, dat_full_ds_top2NDVI_highestVZA)

# del nan_inds_NDVI, dat_full_ds_top2NDVI_lowestVZA, dat_full_ds_top2NDVI_highestVZA
# gc.collect() # manual garbage collection


# clean up the dataset for saving
vars_to_drop = ['time', 'timeobs', 'Orbits', 'NDVI_rank', 'VZA_rank']
for var_to_drop in vars_to_drop:
    if var_to_drop in composite_ds.data_vars:
        composite_ds = composite_ds.drop(var_to_drop)
composite_dayofyear = (composite_period * 16) + 1
dat_date = dt.strptime(f"{composite_year}{composite_dayofyear:03d}", "%Y%j")
composite_ds = composite_ds.expand_dims(time = [dat_date])

composite_ds = composite_ds.assign(CCI=(composite_ds["Sur_refl11"] - composite_ds["Sur_refl1"])/
                           (composite_ds["Sur_refl11"] + composite_ds["Sur_refl1"]))
# composite_ds = composite_ds.assign(NDMI=(composite_ds["Sur_refl6"] - composite_ds["Sur_refl1"])/
#                            (composite_ds["Sur_refl6"] + composite_ds["Sur_refl1"]))
composite_ds = composite_ds.assign(NDVI=(composite_ds["Sur_refl2"] - composite_ds["Sur_refl1"])/
                           (composite_ds["Sur_refl2"] + composite_ds["Sur_refl1"]))
# composite_ds = composite_ds.assign(NDSI=(composite_ds["Sur_refl4"] - composite_ds["Sur_refl6"])/
#                            (composite_ds["Sur_refl4"] + composite_ds["Sur_refl6"]))
# composite_ds = composite_ds.assign(NIRv=(composite_ds["Sur_refl2"] - composite_ds["Sur_refl1"])*composite_ds["Sur_refl2"]/
#                            (composite_ds["Sur_refl2"] + composite_ds["Sur_refl1"]))


# #### Propogate reported uncertainty of band reflectance to vegetation indices
# #### (Miura et al 2000, IEEE Trans Geosci Remote Sens) https://doi.org/10.1109/36.843034

# dNDVI_dB1 = -2*composite_ds.Sur_refl2/((composite_ds.Sur_refl2 + composite_ds.Sur_refl1)**2)
# dNDVI_dB2 = 2*composite_ds.Sur_refl1/((composite_ds.Sur_refl2 + composite_ds.Sur_refl1)**2)
# NDVI_uncertainty = np.sqrt((dNDVI_dB1**2 * composite_ds.Sigma_BRFn1**2) + (dNDVI_dB2**2 * composite_ds.Sigma_BRFn2**2))

# dCCI_dB1 = -2*composite_ds.Sur_refl11/((composite_ds.Sur_refl11 + composite_ds.Sur_refl1)**2)
# dCCI_dB11 = 2*composite_ds.Sur_refl1/((composite_ds.Sur_refl11 + composite_ds.Sur_refl1)**2)
# CCI_uncertainty = np.sqrt((dCCI_dB1**2 * composite_ds.Sigma_BRFn1**2) + (dCCI_dB11**2 * composite_ds.Sigma_BRFn1**2))

# composite_ds = composite_ds.assign(sigma_NDVI = NDVI_uncertainty)
# composite_ds = composite_ds.assign(sigma_CCI = CCI_uncertainty)


# test_plot(composite_ds.NDVI, f'{str(dat_date.date())} - Composite NDVI', 'NDVI', None,
#                     #  f'test_NDVI/{Path(file).stem}_1km_NDVI_Orbit{Orbit}.png',
#                      plot_min = 0, plot_max = 1)

# test_plot(composite_ds.sigma_NDVI, f'{str(dat_date.date())} - Composite NDVI Uncertainty', 'sigma_NDVI', None,
#                     #  f'test_NDVI/{Path(file).stem}_1km_NDVI_Orbit{Orbit}.png',
#                      plot_min = 0, plot_max = 1)


# test_plot(composite_ds.CCI, f'{str(dat_date.date())} - Composite CCI', 'CCI', None,
#                     #  f'test_CCI/{Path(file).stem}_1km_CCI_Orbit{Orbit}.png',
#                      plot_min = -0.3, plot_max = 0.3)

# test_plot(composite_ds.sigma_CCI, f'{str(dat_date.date())} - Composite CCI Uncertainty', 'sigma_CCI', None,
#                     #  f'test_CCI/{Path(file).stem}_1km_CCI_Orbit{Orbit}.png',
#                      plot_min = 0, plot_max = 0.6)

#### Basic VI usefulness flag:
# 0 = no data
# 1 = Highest Quality
# 2 = Data passed QC, but snow cover detected from MAIAC surface classification algorithm
# 3 = Data passed QC, but value removed from outlier filter algorithm (next step)
# 4 = Value derived from climatology gap filling algorithm

VI_usefulness_flag = xr.where(composite_ds.NDVI.notnull(), 0, 0)  # Initialize to 0 where data exists, 0 (missing) where null
VI_usefulness_flag = xr.where((composite_ds.SnowIce == 0) & composite_ds.NDVI.notnull(), 1, VI_usefulness_flag)  # Set to 1 if highest quality
VI_usefulness_flag = xr.where((composite_ds.SnowIce == 1) & composite_ds.NDVI.notnull(), 2, VI_usefulness_flag)  # Set to 2 if snow cover detected
# Assign remainder in future steps

composite_ds = composite_ds.assign(QA_flag = VI_usefulness_flag)

test_plot(composite_ds.QA_flag == 0, f'{str(dat_date.date())} - No Data', 'No Data Flag (0/1)', None,
                    #  f'test_CCI/{Path(file).stem}_1km_CCI_Orbit{Orbit}.png',
                     plot_min = 0, plot_max = 1)

test_plot(composite_ds.QA_flag == 1, f'{str(dat_date.date())} - Highest Quality', 'Highest Quality Flag (0/1)', None,
                    #  f'test_CCI/{Path(file).stem}_1km_CCI_Orbit{Orbit}.png',
                     plot_min = 0, plot_max = 1)

test_plot(composite_ds.QA_flag == 2, f'{str(dat_date.date())} - Snow cover detected', 'Snow Ice Flag (0/1)', None,
                    #  f'test_CCI/{Path(file).stem}_1km_CCI_Orbit{Orbit}.png',
                     plot_min = 0, plot_max = 1)

# you can drop these vars from the dataset now, they aren't needed
drop_vars = ["CloudFree", "LooseCloud", "LowAOD", "ClearAdj", "NotSurround", "SnowIce", "all_nan_mask",  "cosVZA"] # drop the QA mask variables since we won't need them anymore
composite_ds = composite_ds.drop_vars(drop_vars)

# composite_ds['time'] = pd.DatetimeIndex(composite_ds['time'].values)
# composite_ds.time.encoding['units'] = f'Days since 2000-01-01'

# # Define attributes for all variables being saved in netcdf
# composite_ds.CCI.attrs['long_name'] = 'Chlorophyll Carotenoid Index'
# composite_ds.NDVI.attrs['long_name'] = 'Normalized Difference Vegetation Index'
# composite_ds.NDMI.attrs['long_name'] = 'Normalized Difference Moisture Index'
# composite_ds.NDSI.attrs['long_name'] = 'Normalized Difference Snow Index'
# composite_ds.NIRv.attrs['long_name'] = 'Near-infrared Reflectance of Vegetation'
# # composite_ds.kNDVI.attrs['long_name'] = 'kernel Normalized Difference Vegetation Index'

# # include formulas
# composite_ds.CCI.attrs['formula'] = '(B11 - B1)/(B11 + B1)'
# composite_ds.NDVI.attrs['formula'] = '(B2 - B1)/(B2 + B1)'
# composite_ds.NDMI.attrs['formula'] = '(B6 - B1)/(B6 + B1)'
# composite_ds.NDSI.attrs['formula'] = '(B4 - B6)/(B4 + B6)'
# composite_ds.NIRv.attrs['formula'] = 'NDVI*B2'
# # composite_ds.kNDVI.attrs['formula'] = 'tanh(NDVI^2)'

# # define global attributes
# composite_ds.attrs['title'] = 'MODIS MCD19A1.061 MAIAC Vegetation Indices calculated from surface reflectance - 16-day composite using Constrained View angle - Maximum Value Composite (CV-MVC) algorithm'
# composite_ds.attrs['date_string'] = str(dat_date.date())
# composite_ds.attrs['Comment1'] = 'Original files downloaded from https://e4ftl01.cr.usgs.gov/MOTA/MCD19A1.061/'
# composite_ds.attrs['Comment2'] = 'hdf files converted to netcdf4 using HDF-EOS conversion tool (https://www.hdfeos.org/software/h4toh5.php)'
# composite_ds.attrs['Comment3'] = 'regridded from multi-dimensional x,y coordinates to 0.01° lat, lon spacing'
# composite_ds.attrs['Citation'] = 'Lyapustin, A., Wang, Y. (2022). MODIS/Terra+Aqua Land Surface BRF Daily L2G Global 500m and 1km SIN Grid V061 [Data set]. NASA EOSDIS Land Processes Distributed Active Archive Center. Accessed from https://doi.org/10.5067/MODIS/MCD19A1.061'
# composite_ds.attrs['Processed_by'] = 'Lewis Kunik, lewis.kunik@utah.edu'
# composite_ds.attrs['Date_created'] = str(dt.today().date())

# # Finally, save to netcdf
# composite_ds.to_netcdf(path=outfile, format="NETCDF4")
# composite_ds.close()



# test_plot(dat_da = composite_ds.NDVI, 
#             plot_title=str(filedate), zlab='NDVI',
#             plot_filename=f'{plot_dir}/NDVI/MCD19A1.h09v04.061_{str(filedate)}_1km_NDVI_CV-MVC_composite.png',
#             plot_min = 0, plot_max = 1.2)

# test_plot(dat_da = composite_ds.CCI, 
#             plot_title=str(filedate), zlab='CCI',
#             plot_filename=f'{plot_dir}/CCI/MCD19A1.h09v04.061_{str(filedate)}_1km_CCI_CV-MVC_composite.png',
#             plot_min = -0.3, plot_max = 0.5)

end_total = time.time()
total_elapsed = timedelta(seconds=end_total-start_total)
print(f'File saved. Total elapsed time = {td_min(total_elapsed)} mins, {td_sec(total_elapsed)} seconds')

del composite_ds # remove large variables from environment to free up space
gc.collect()

# %%
