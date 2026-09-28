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
import rasterio
import xarray as xr  # for multi dimensional data
# import rioxarray as rxr
import pandas as pd
import xesmf as xe
import struct
from rasterio.enums import Resampling
import re

# shapefile/geospatial packages

# time and date packages
import time
from datetime import datetime, timedelta
from datetime import datetime as dt  # date time library

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
MOD09_basedir = "/uufs/chpc.utah.edu/common/home/lin-group23/ltk/MODIS/Reflectance_MOD09/from_GEE/"
out_basedir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/MODIS/Reflectance_MOD09/CV-MVC/'




#########################################
# Define Global functions
#########################################

def floor(value, decimal_digits):
  scale = (10**decimal_digits)
  return  np.floor(value * scale)/scale

def ceil(value, decimal_digits):
  scale = (10**decimal_digits)
  return  np.ceil(value * scale)/scale

def binary32(num):
    return ''.join('{:0>32b}'.format(c) for c in struct.pack('!f', num))

binary32_v = np.vectorize(binary32)

def get_bits(str, start_bit, end_bit):
   str_split = [*str][start_bit:end_bit+1]
   return ''.join(str_split)

get_bits_v = np.vectorize(get_bits)


def get_bits_str(str, start_bit, end_bit):
    if start_bit == 0:
        str_split = [*str][-(end_bit+1):]
    else:
        str_split = [*str][-(end_bit+1):-(start_bit)]
    return ''.join(str_split)

get_bits_str_v = np.vectorize(get_bits_str)


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




def extract_dates_from_bands(band_names):
    """
    Extract dates from MODIS band names
    
    Args:
        band_names: list of band names from the GeoTIFF
    
    Returns:
        list of datetime objects
    """
    dates = []
    
    for i, band_name in enumerate(band_names):
        # Handle None or empty band names
        if band_name is None or band_name == '':
            dates.append(None)
            continue
            
        # Convert to string if not already
        band_name = str(band_name)
        
        # MODIS band names typically contain date info
        # Try to extract date patterns like YYYY_MM_dd
        date_match = re.search(r'(\d{4})_(\d{2})_(\d{2})', band_name)
        if date_match:
            year, month, day = date_match.groups()
            try:
                date = datetime(int(year), int(month), int(day))
                dates.append(date)
                continue
            except:
                pass
        
        # Try day of year format YYYY_DDD or YYYYDDD
        doy_match = re.search(r'(\d{4})_?(\d{3})', band_name)
        if doy_match:
            year, doy = doy_match.groups()
            try:
                date = datetime(int(year), 1, 1) + timedelta(days=int(doy)-1)
                dates.append(date)
                continue
            except:
                pass
        
        # Try simple YYYY format and estimate based on band index
        year_match = re.search(r'(\d{4})', band_name)
        if year_match:
            year = int(year_match.group(1))
            # Estimate date based on band index (assuming 8-day composites, ~46 per year)
            days_offset = (i % 46) * 8
            try:
                date = dt(year, 1, 1) + timedelta(days=days_offset)
                dates.append(date)
                continue
            except:
                pass
        
        # If all else fails, use band index as fallback
        dates.append(None)
    
    return dates
#########################################
# Define Global Variables and constants
#########################################
#%%

# view zenith angle above which we will exclude obs for "case 2", based on "HIGHVIEW" flag here
# https://landweb.modaps.eosdis.nasa.gov/NPP/forPage/VIIRS_NPP_Surf_Refl_UserGuide_v1.1.pdf
# VZA_thresh = 45 

min_Valid_obs_for_composite = 4

#########################################
# Begin main
#########################################
#%%

def main():

    # mark start time to keep track of elapsed
    start_total = time.time()

    site_name = sys.argv[1] #'US-NR1' 
    composite_year = int(sys.argv[2]) #2017 # command line argument
    composite_period = int(sys.argv[3]) # 14 # command line argument
        
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
    QC_option = sys.argv[4] # command line argument - options are 'CloudFree', 'CloudAOD', 'Cloud', 'AOD', or 'None'

    dat_dir = os.path.join(MOD09_basedir, site_name)
    MOD09GA_files = sorted(glob.glob(f'{dat_dir}/MOD09GA*.tif')) #
    MYD09GA_files = sorted(glob.glob(f'{dat_dir}/MYD09GA*.tif')) #
    MODOCGA_files = sorted(glob.glob(f'{dat_dir}/MODOCGA*.tif')) #
    MYDOCGA_files = sorted(glob.glob(f'{dat_dir}/MYDOCGA*.tif')) #
    out_dir = os.path.join(out_basedir, f'QCfilt_{QC_option}', site_name)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    outfile = os.path.join(out_dir, f'MOD09_{site_name}_{QC_option}_{composite_year}-{str(composite_period).zfill(2)}.nc')

    if (len(MOD09GA_files) == 0) | (len(MYD09GA_files) == 0) | (len(MODOCGA_files) == 0) | (len(MYDOCGA_files) == 0):
        print(f'One or more sets of MOD09 files not found for {site_name}, {composite_year}-{str(composite_period).zfill(2)}, skipping processing')
        sys.exit(0)

    # plot_dir = os.path.join(plot_basedir, f'QCfilt_{QC_option}', site_name)
    # Path(plot_dir).mkdir(parents=True, exist_ok=True)

    DOY_composite_starts = np.arange(1, 366, 16)
    MOD09GA_filedates = [dt.strptime(os.path.basename(file)[-14:-4], '%Y-%m-%d').date() for file in MOD09GA_files]
    MOD09GA_filedoys = [int(date.strftime('%j')) for date in MOD09GA_filedates]
    MOD09GA_file_composite_periods = np.searchsorted(DOY_composite_starts, MOD09GA_filedoys, side='right') - 1

    MYD09GA_filedates = [dt.strptime(os.path.basename(file)[-14:-4], '%Y-%m-%d').date() for file in MYD09GA_files]
    MYD09GA_filedoys = [int(date.strftime('%j')) for date in MYD09GA_filedates]
    MYD09GA_file_composite_periods = np.searchsorted(DOY_composite_starts, MYD09GA_filedoys, side='right') - 1

    MODOCGA_filedates = [dt.strptime(os.path.basename(file)[-14:-4], '%Y-%m-%d').date() for file in MODOCGA_files]
    MODOCGA_filedoys = [int(date.strftime('%j')) for date in MODOCGA_filedates]
    MODOCGA_file_composite_periods = np.searchsorted(DOY_composite_starts, MODOCGA_filedoys, side='right') - 1

    MYDOCGA_filedates = [dt.strptime(os.path.basename(file)[-14:-4], '%Y-%m-%d').date() for file in MYDOCGA_files]
    MYDOCGA_filedoys = [int(date.strftime('%j')) for date in MYDOCGA_filedates]
    MYDOCGA_file_composite_periods = np.searchsorted(DOY_composite_starts, MYDOCGA_filedoys, side='right') - 1

    if os.path.exists(outfile):
        print(f'Output file {os.path.basename(outfile)} already exists, skipping processing for {composite_year}-{str(composite_period).zfill(2)}')
        sys.exit(0)


    # Band names for MOD09GA (sur_refl_b01 through sur_refl_b07)
    band_names = ['sur_refl_b01', 'sur_refl_b02', 'SensorZenith', 'state_1km', 'QC_500m'] # HARD CODED FROM PREV SCRIPT
    band_scale_factors = [0.0001, 0.0001, 0.01, 1, 1] # https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD09GA

    band_names_ocga = ['sur_refl_b11', 'sur_refl_b12', 'QC_b8_15_1km']
    band_scale_factors_ocga = [0.0001, 0.0001, 1] # https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD09GA


    iYYMM_MOD = [ i for i in range(len(MOD09GA_filedates)) if ((MOD09GA_filedates[i].year == composite_year) &
                                                    (MOD09GA_file_composite_periods[i] == composite_period)) ]
    dat_xrs = []

    if len(iYYMM_MOD) == 0:
        print(f'No files found for {composite_year}-{str(composite_period).zfill(2)}, skipping processing')
        sys.exit(0)

    tot_gridcells_16day = 0
    tot_orig_nonNA_16day = 0

    ### Produce XESMF regridder objects for 1km and 5km data before looping through files
    isample = 0
    sample_tif_file = MOD09GA_files[iYYMM_MOD[isample]]

    # Start obs counter
    iobs = 0

    ## Add TERRA obs

    num_iter = 2
    ii = iYYMM_MOD[2] 
    for num_iter, ii in enumerate(iYYMM_MOD):
        
        mod09ga_file = MOD09GA_files[ii]
        filedate = MOD09GA_filedates[ii]
        if os.stat(mod09ga_file).st_size == 0:
            print(f'    File {os.path.basename(mod09ga_file)} is empty, skipping')
            continue
        # print(f'    loading file: {os.path.basename(mod09ga_file)} - 1km data')
        start_file = time.time()


        # Here is where processing of each file needs to happen - load into xarray object (will have no spatial dimension, maybe not a time dim)
        # Load MOD09GA data (bands 1-7)
        with rasterio.open(mod09ga_file) as src:

            # print(f"Dimensions: {src.height} x {src.width} pixels")
            # print(f"Number of bands: {src.count}")
            # print(f"Resolution: {src.res}")
            # print(f"Bounds: {src.bounds}")
            # print(f"CRS: {src.crs}")
            
            # # Read the data
            # data = src.read()
            # print(f"Data shape: {data.shape}")
            
            # # Check QC_500m values
            # qc_data = data[2, :, :]  # QC_500m is band 3
            # print(f"\nQC_500m shape: {qc_data.shape}")
            # print(f"QC_500m values:\n{qc_data}")
            
            # # Check SensorZenith
            # sz_data = data[3, :, :]  # SensorZenith is band 4
            # print(f"\nSensorZenith shape: {sz_data.shape}")
            # print(f"SensorZenith values:\n{sz_data}")

            # Read all bands as numpy array
            refl_data = src.read()  # shape: (7, height, width)
            
            # Get georeferencing info
            transform = src.transform
            crs = src.crs
            
            # Create coordinate arrays
            height, width = src.height, src.width
            x_coords = np.arange(width) * transform[0] + transform[2]
            y_coords = np.arange(height) * transform[4] + transform[5]
            
            # Create xarray Dataset
            data_vars = {}
            for i, band_name in enumerate(band_names):
                # Apply scale factor
                data_vars[band_name] = (['y', 'x'], refl_data[i, :, :] * band_scale_factors[i])

            ds_mod_GA = xr.Dataset(
                data_vars=data_vars,
                coords={
                    'x': x_coords,
                    'y': y_coords,
                    'time': filedate
                }
            )
            ds_mod_GA = ds_mod_GA.squeeze(['x', 'y'])
            
            # State QC (clouds, etc)
            state_QC_bits = format(int(ds_mod_GA['state_1km']), '016b')
            cloud_state = get_bits_str_v(state_QC_bits, 0, 1)
            cloud_shadow = get_bits_str_v(state_QC_bits, 2, 2)
            # land_water = get_bits_str_v(state_QC_bits, 3, 5)
            AOD_state = get_bits_str_v(state_QC_bits, 6, 7)
            # cirrus_state = get_bits_str_v(state_QC_bits, 8, 9)
            cloud_Internal_state = get_bits_str_v(state_QC_bits, 10, 10) # 0 if clear, 1 if cloudy
            fire_Internal_state = get_bits_str_v(state_QC_bits, 11, 11)
            snow_ice_MOD35_state = get_bits_str_v(state_QC_bits, 12, 12)
            adjacency_bits = get_bits_str_v(state_QC_bits, 13, 13)
            snow_ice_Internal_state = get_bits_str_v(state_QC_bits, 15, 15)
       

            CloudFree = (np.isin(cloud_state, ['00', '11']) & (cloud_shadow == '0')) & (cloud_Internal_state == '0')
            LooseCloud = np.isin(cloud_state, ['00', '10', '11']) & (cloud_Internal_state == '0')
            LowAOD = np.isin(AOD_state, ['00', '01', '10'])
            ClearAdj = (adjacency_bits == '0')
            fire = (fire_Internal_state == '1') # 0 if no fire, 1 if fire
            snow_ice = (snow_ice_MOD35_state == '1') | (snow_ice_Internal_state == '1')  # 0 if no snow/ice, 1 if snow/ice

            # Apply the final QC filter based on function input option
            if QC_option == 'CloudFree_LowAOD_ClearAdj':
                QCpass = CloudFree & LowAOD & ClearAdj
            elif QC_option == 'CloudFree_LowAOD_NotSurround':
                QCpass = CloudFree & LowAOD
            elif QC_option == 'CloudFree_LowAOD_AllAdj':
                QCpass = CloudFree & LowAOD
            elif QC_option == 'CloudFree_AllAOD_ClearAdj':
                QCpass = CloudFree & ClearAdj 
            elif QC_option == 'CloudFree_AllAOD_NotSurround':
                QCpass = CloudFree & LowAOD 
            elif QC_option == 'CloudFree_AllAOD_AllAdj':
                QCpass = CloudFree
            elif QC_option == 'LooseCloud_LowAOD_ClearAdj':
                QCpass = LooseCloud & LowAOD & ClearAdj
            elif QC_option == 'LooseCloud_LowAOD_NotSurround':
                QCpass = LooseCloud & LowAOD
            elif QC_option == 'LooseCloud_LowAOD_AllAdj':
                QCpass = LooseCloud & LowAOD
            elif QC_option == 'LooseCloud_AllAOD_ClearAdj':
                QCpass = LooseCloud & ClearAdj
            elif QC_option == 'LooseCloud_AllAOD_NotSurround':
                QCpass = LooseCloud
            elif QC_option == 'LooseCloud_AllAOD_AllAdj':
                QCpass = LooseCloud
            elif QC_option == 'AllCloud_LowAOD_ClearAdj':
                QCpass = LowAOD & ClearAdj
            elif QC_option == 'AllCloud_LowAOD_NotSurround':
                QCpass = LowAOD
            elif QC_option == 'AllCloud_LowAOD_AllAdj':
                QCpass = LowAOD
            elif QC_option == 'AllCloud_AllAOD_ClearAdj':
                QCpass = ClearAdj
            elif QC_option == 'AllCloud_AllAOD_NotSurround':
                QCpass = True
            elif QC_option == 'AllCloud_AllAOD_AllAdj':
                QCpass = True
            elif QC_option == 'CloudFree': # REMOVE THIS AFTER FILLING h09v02 FOR THE WILKES POSTER
                QCpass = CloudFree
            else:
                raise ValueError(f'Invalid QC_option: {QC_option}')

            # If snow/ice detected on the surface, set to nan
            if snow_ice:
                QCpass = False
            
            if fire:
                QCpass = False

            if not QCpass:
                ds_mod_GA['sur_refl_b01'] = np.nan
                ds_mod_GA['sur_refl_b02'] = np.nan

            # Mandatory QC
            Mandatory_QC_bits = format(int(ds_mod_GA['QC_500m']), '032b')
            MODLAND_QC_bits = get_bits_str_v(Mandatory_QC_bits, 0, 1)
            QC_Band1 = get_bits_str_v(Mandatory_QC_bits, 2, 5)
            QC_Band2 = get_bits_str_v(Mandatory_QC_bits, 6, 9)

            if np.isin(MODLAND_QC_bits, ['10', '11']):
                ds_mod_GA['sur_refl_b01'] = np.nan
                ds_mod_GA['sur_refl_b02'] = np.nan

            if QC_Band1 != '0000':
                ds_mod_GA['sur_refl_b01'] = np.nan
            if QC_Band2 != '0000':
                ds_mod_GA['sur_refl_b02'] = np.nan

            if ds_mod_GA['sur_refl_b01'] < 0:
                ds_mod_GA['sur_refl_b01'] = np.nan
            if ds_mod_GA['sur_refl_b02'] < 0:
                ds_mod_GA['sur_refl_b02'] = np.nan

            # Expand time dimension
            ds_mod_GA = ds_mod_GA.expand_dims(obs = [iobs])
        
        # Now load corresponding MODOCGA file (bands 8-12) for the same date
        # Find the matching MODOCGA file
        modocga_file = None
        for jj in range(len(MODOCGA_filedates)):
            if MODOCGA_filedates[jj] == filedate:
                modocga_file = MODOCGA_files[jj]
                break
        
        if modocga_file is None:
            print(f'    Warning: No matching MODOCGA file found for {filedate}, skipping this date')
            continue
        
        if os.stat(modocga_file).st_size == 0:
            print(f'    File {os.path.basename(modocga_file)} is empty, skipping')
            continue
        
        print(f'    loading corresponding MODOCGA file: {os.path.basename(modocga_file)}')
        
        # Load MODOCGA data (bands 8-12)
        with rasterio.open(modocga_file) as src:
            # Read all bands
            refl_data_ocga = src.read()  # shape: (5, height, width)
                    
            # Create data variables
            data_vars_ocga = {}
            for i, band_name in enumerate(band_names_ocga):
                # Apply scale factor
                data_vars_ocga[band_name] = (['y', 'x'], refl_data_ocga[i, :, :] * band_scale_factors_ocga[i])
            
            ds_mod_OCGA = xr.Dataset(
                data_vars=data_vars_ocga,
                coords={
                    'x': x_coords,
                    'y': y_coords,
                    'time': filedate
                }
            )
            ds_mod_OCGA = ds_mod_OCGA.squeeze(['x', 'y'])

            if not QCpass:
                ds_mod_OCGA['sur_refl_b11'] = np.nan
                ds_mod_OCGA['sur_refl_b12'] = np.nan

            if ds_mod_OCGA['sur_refl_b11'] < 0:
                ds_mod_OCGA['sur_refl_b11'] = np.nan
            if ds_mod_OCGA['sur_refl_b12'] < 0:
                ds_mod_OCGA['sur_refl_b12'] = np.nan                         
            # Expand time dimension
            ds_mod_OCGA = ds_mod_OCGA.expand_dims(obs = [iobs])
            
        iobs += 1
        
        # Merge the two datasets
        mod_ds_combined = xr.merge([ds_mod_GA, ds_mod_OCGA])
        
        # Append to list
        dat_xrs.append(mod_ds_combined)


    ### add AQUA obs


    iYYMM_MYD = [ i for i in range(len(MYD09GA_filedates)) if ((MYD09GA_filedates[i].year == composite_year) &
                                                    (MYD09GA_file_composite_periods[i] == composite_period)) ]

    if len(iYYMM_MYD) == 0:
        print(f'No files found for {composite_year}-{str(composite_period).zfill(2)}, skipping processing')
        # sys.exit(0)


    num_iter = 2
    ii = iYYMM_MYD[2] 
    for num_iter, ii in enumerate(iYYMM_MYD):
        
        myd09ga_file = MYD09GA_files[ii]
        filedate = MYD09GA_filedates[ii]
        if os.stat(myd09ga_file).st_size == 0:
            print(f'    File {os.path.basename(myd09ga_file)} is empty, skipping')
            continue
        # print(f'    loading file: {os.path.basename(myd09ga_file)} - 1km data')
        start_file = time.time()


        # Band names for MOD09GA (sur_refl_b01 through sur_refl_b07)
        band_names = ['sur_refl_b01', 'sur_refl_b02', 'SensorZenith', 'state_1km', 'QC_500m'] # HARD CODED FROM PREV SCRIPT
        band_scale_factors = [0.0001, 0.0001, 0.01, 1, 1] # https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD09GA
        
        # Here is where processing of each file needs to happen - load into xarray object (will have no spatial dimension, maybe not a time dim)
        # Load MOD09GA data (bands 1-7)
        with rasterio.open(myd09ga_file) as src:

            # print(f"Dimensions: {src.height} x {src.width} pixels")
            # print(f"Number of bands: {src.count}")
            # print(f"Resolution: {src.res}")
            # print(f"Bounds: {src.bounds}")
            # print(f"CRS: {src.crs}")
            
            # # Read the data
            # data = src.read()
            # print(f"Data shape: {data.shape}")
            
            # # Check QC_500m values
            # qc_data = data[2, :, :]  # QC_500m is band 3
            # print(f"\nQC_500m shape: {qc_data.shape}")
            # print(f"QC_500m values:\n{qc_data}")
            
            # # Check SensorZenith
            # sz_data = data[3, :, :]  # SensorZenith is band 4
            # print(f"\nSensorZenith shape: {sz_data.shape}")
            # print(f"SensorZenith values:\n{sz_data}")

            # Read all bands as numpy array
            refl_data = src.read()  # shape: (7, height, width)
            
            # Get georeferencing info
            transform = src.transform
            crs = src.crs
            
            # Create coordinate arrays
            height, width = src.height, src.width
            x_coords = np.arange(width) * transform[0] + transform[2]
            y_coords = np.arange(height) * transform[4] + transform[5]
            
            # Create xarray Dataset
            data_vars = {}
            for i, band_name in enumerate(band_names):
                # Apply scale factor
                data_vars[band_name] = (['y', 'x'], refl_data[i, :, :] * band_scale_factors[i])

            ds_myd_GA = xr.Dataset(
                data_vars=data_vars,
                coords={
                    'x': x_coords,
                    'y': y_coords,
                    'time': filedate
                }
            )
            ds_myd_GA = ds_myd_GA.squeeze(['x', 'y'])
            
            # State QC (clouds, etc)
            state_QC_bits = format(int(ds_myd_GA['state_1km']), '016b')
            cloud_state = get_bits_str_v(state_QC_bits, 0, 1)
            cloud_shadow = get_bits_str_v(state_QC_bits, 2, 2)
            # land_water = get_bits_str_v(state_QC_bits, 3, 5)
            AOD_state = get_bits_str_v(state_QC_bits, 6, 7)
            cirrus_state = get_bits_str_v(state_QC_bits, 8, 9)
            cloud_Internal_state = get_bits_str_v(state_QC_bits, 10, 10) # 0 if clear, 1 if cloudy
            fire_Internal_state = get_bits_str_v(state_QC_bits, 11, 11)
            snow_ice_MOD35_state = get_bits_str_v(state_QC_bits, 12, 12)
            adjacency_bits = get_bits_str_v(state_QC_bits, 13, 13)
            snow_ice_Internal_state = get_bits_str_v(state_QC_bits, 15, 15)
            

            CloudFree = (np.isin(cloud_state, ['00', '11']) & (cloud_shadow == '0')) & (cloud_Internal_state == '0') & (np.isin(cirrus_state, ['00', '01']))
            LooseCloud = np.isin(cloud_state, ['00', '10', '11']) & (cloud_Internal_state == '0')
            LowAOD = np.isin(AOD_state, ['00', '01', '10'])
            ClearAdj = (adjacency_bits == '0')
            fire = (fire_Internal_state == '1') # 0 if no fire, 1 if fire
            snow_ice = (snow_ice_MOD35_state == '1') | (snow_ice_Internal_state == '1')  # 0 if no snow/ice, 1 if snow/ice

            # Apply the final QC filter based on function input option
            if QC_option == 'CloudFree_LowAOD_ClearAdj':
                QCpass = CloudFree & LowAOD & ClearAdj
            elif QC_option == 'CloudFree_LowAOD_NotSurround':
                QCpass = CloudFree & LowAOD
            elif QC_option == 'CloudFree_LowAOD_AllAdj':
                QCpass = CloudFree & LowAOD
            elif QC_option == 'CloudFree_AllAOD_ClearAdj':
                QCpass = CloudFree & ClearAdj 
            elif QC_option == 'CloudFree_AllAOD_NotSurround':
                QCpass = CloudFree & LowAOD 
            elif QC_option == 'CloudFree_AllAOD_AllAdj':
                QCpass = CloudFree
            elif QC_option == 'LooseCloud_LowAOD_ClearAdj':
                QCpass = LooseCloud & LowAOD & ClearAdj
            elif QC_option == 'LooseCloud_LowAOD_NotSurround':
                QCpass = LooseCloud & LowAOD
            elif QC_option == 'LooseCloud_LowAOD_AllAdj':
                QCpass = LooseCloud & LowAOD
            elif QC_option == 'LooseCloud_AllAOD_ClearAdj':
                QCpass = LooseCloud & ClearAdj
            elif QC_option == 'LooseCloud_AllAOD_NotSurround':
                QCpass = LooseCloud
            elif QC_option == 'LooseCloud_AllAOD_AllAdj':
                QCpass = LooseCloud
            elif QC_option == 'AllCloud_LowAOD_ClearAdj':
                QCpass = LowAOD & ClearAdj
            elif QC_option == 'AllCloud_LowAOD_NotSurround':
                QCpass = LowAOD
            elif QC_option == 'AllCloud_LowAOD_AllAdj':
                QCpass = LowAOD
            elif QC_option == 'AllCloud_AllAOD_ClearAdj':
                QCpass = ClearAdj
            elif QC_option == 'AllCloud_AllAOD_NotSurround':
                QCpass = True
            elif QC_option == 'AllCloud_AllAOD_AllAdj':
                QCpass = True
            elif QC_option == 'CloudFree': # REMOVE THIS AFTER FILLING h09v02 FOR THE WILKES POSTER
                QCpass = CloudFree
            else:
                raise ValueError(f'Invalid QC_option: {QC_option}')


            # If snow/ice detected on the surface, set to nan
            if snow_ice:
                QCpass = False
            
            if fire:
                QCpass = False

            if not QCpass:
                ds_myd_GA['sur_refl_b01'] = np.nan
                ds_myd_GA['sur_refl_b02'] = np.nan

            # Mandatory QC
            Mandatory_QC_bits = format(int(ds_myd_GA['QC_500m']), '032b')
            MODLAND_QC_bits = get_bits_str_v(Mandatory_QC_bits, 0, 1)
            QC_Band1 = get_bits_str_v(Mandatory_QC_bits, 2, 5)
            QC_Band2 = get_bits_str_v(Mandatory_QC_bits, 6, 9)

            if np.isin(MODLAND_QC_bits, ['10', '11']):
                ds_myd_GA['sur_refl_b01'] = np.nan
                ds_myd_GA['sur_refl_b02'] = np.nan

            if QC_Band1 != '0000':
                ds_myd_GA['sur_refl_b01'] = np.nan
            if QC_Band2 != '0000':
                ds_myd_GA['sur_refl_b02'] = np.nan

            if ds_myd_GA['sur_refl_b01'] < 0:
                ds_myd_GA['sur_refl_b01'] = np.nan
            if ds_myd_GA['sur_refl_b02'] < 0:
                ds_myd_GA['sur_refl_b02'] = np.nan
                # Expand time dimension
            ds_myd_GA = ds_myd_GA.expand_dims(obs = [iobs])
        
        # Now load corresponding MODOCGA file (bands 8-12) for the same date
        # Find the matching MODOCGA file
        mydocga_file = None
        for jj in range(len(MODOCGA_filedates)):
            if MODOCGA_filedates[jj] == filedate:
                mydocga_file = MODOCGA_files[jj]
                break
        
        if mydocga_file is None:
            print(f'    Warning: No matching MODOCGA file found for {filedate}, skipping this date')
            continue
        
        if os.stat(mydocga_file).st_size == 0:
            print(f'    File {os.path.basename(mydocga_file)} is empty, skipping')
            continue
        
        print(f'    loading corresponding MODOCGA file: {os.path.basename(mydocga_file)}')
        
        # Load MODOCGA data (bands 8-12)
        with rasterio.open(mydocga_file) as src:
            # Read all bands
            refl_data_ocga = src.read()  # shape: (5, height, width)
            
            # Band names for MODOCGA (sur_refl_b08 through sur_refl_b12)
            
            
            # Create data variables
            data_vars_ocga = {}
            for i, band_name in enumerate(band_names_ocga):
                # Apply scale factor
                data_vars_ocga[band_name] = (['y', 'x'], refl_data_ocga[i, :, :] * band_scale_factors_ocga[i])
            
            ds_myd_OCGA = xr.Dataset(
                data_vars=data_vars_ocga,
                coords={
                    'x': x_coords,
                    'y': y_coords,
                    'time': filedate
                }
            )

            ds_myd_OCGA = ds_myd_OCGA.squeeze(['x', 'y'])

            if not QCpass:
                ds_myd_OCGA['sur_refl_b11'] = np.nan
                ds_myd_OCGA['sur_refl_b12'] = np.nan

            if ds_myd_OCGA['sur_refl_b11'] < 0:
                ds_myd_OCGA['sur_refl_b11'] = np.nan
            if ds_myd_OCGA['sur_refl_b12'] < 0:
                ds_myd_OCGA['sur_refl_b12'] = np.nan                            
            # Expand time dimension
            ds_myd_OCGA = ds_myd_OCGA.expand_dims(obs = [iobs])
            
        iobs += 1
        
        # Merge the two datasets
        myd_ds_combined = xr.merge([ds_myd_GA, ds_myd_OCGA])
        
        # Append to list
        dat_xrs.append(myd_ds_combined)

    print(f'Done stacking daily files for {composite_year}-{str(composite_period).zfill(2)}. Concatenating into single dataset...')
    dat_full_ds = xr.concat(dat_xrs, dim = 'obs') # concatenate along time axis


    #####################################
    ### Implement CV-MVC algorithm
    #####################################

    # Calculate NDVI for the CV-MVC algorithm
    dat_full_ds = dat_full_ds.assign(NDVI=(dat_full_ds["sur_refl_b02"] - dat_full_ds["sur_refl_b01"])/
                            (dat_full_ds["sur_refl_b02"] + dat_full_ds["sur_refl_b01"]))

    # print(np.unique(dat_full_ds['NDVI'].values))
    num_valid_obs = np.count_nonzero(~np.isnan(dat_full_ds['NDVI'].values))

    # you can drop these vars from the dataset now, they aren't needed
    drop_vars = ["state_1km", "QC_500m", "QC_b8_15_1km"]  # drop the QA mask variables since we won't need them anymore
    dat_full_ds = dat_full_ds.drop_vars(drop_vars)
    # Check if we have enough valid observations
    if num_valid_obs < min_Valid_obs_for_composite:
        print(f'Insufficient number of valid observations ({num_valid_obs}) for {composite_year}-{str(composite_period).zfill(2)}, filling composite with nans')
        composite_ds = dat_full_ds.isel(obs=0).drop_vars('obs')
        for var in composite_ds.data_vars:
            composite_ds[var] = xr.full_like(composite_ds[var], np.nan)
    else:
        ### Step 1: pixel-wise sort by NDVI
        # define the array from which we want to sort NDVI values
        arr_sort_NDVI = -dat_full_ds.NDVI # NOTE THE NEGATIVE!!! makes it a descending sort
        sort_axis = arr_sort_NDVI.get_axis_num('obs') # this gets the axis over which we want to sort
        sort_inds_NDVI = arr_sort_NDVI.argsort(axis=sort_axis) # perform the sort to get pixel-wise indices of the highest NDVI, 2nd highest NDVI, etc...
        sort_inds_NDVI = sort_inds_NDVI.rename({'obs': 'NDVI_rank'}) # Replace obs dim with ranking dim
        sort_inds_top2NDVI = sort_inds_NDVI.isel(NDVI_rank = [0,1]) # select only the top 2 NDVI indices
        dat_full_ds_top2NDVI = dat_full_ds.isel({'obs': sort_inds_top2NDVI}) # filter the original stacked dataset for these top 2 NDVI indices

        ### Step 2: pixel-wise sort by View angle
        # pipe the NDVI-filtered dataset into this step to sort by VZA values
        arr_sort_VZA = dat_full_ds_top2NDVI.SensorZenith # NOTE There is NO negative, makes it an ascending sort
        sort_inds_VZA = arr_sort_VZA.argsort(axis=sort_axis) # perform sort to get pixel-wise indices of lowest and 2nd lowest VZA (NOTE sort_axis should be the same axis as with NDVI.)
        sort_inds_VZA = sort_inds_VZA.rename({'NDVI_rank': 'VZA_rank'}) # replace NDVI_rank dim with VZA_rank
        sort_inds_lowVZA = sort_inds_VZA.isel(VZA_rank = 0) # get the index of the lowest VZA
        sort_inds_highVZA = sort_inds_VZA.isel(VZA_rank = 1) # get the other index
        dat_full_ds_top2NDVI_lowestVZA = dat_full_ds_top2NDVI.isel({'NDVI_rank': sort_inds_lowVZA}) # Using the NDVI-filtered stacked dataset, filter for the lowest VZA
        dat_full_ds_top2NDVI_highestVZA = dat_full_ds_top2NDVI.isel({'NDVI_rank': sort_inds_highVZA}) # For any pixels 

        ### Step 3: pixel-wise check for <2 valid NDVI vals
        # if the resulting NDVI value is nan, use the values of the next smallest VZA.
        # (If there was only 1 valid NDVI value to begin, this will recover that value)
        nan_inds_NDVI = np.isnan(dat_full_ds_top2NDVI_lowestVZA.NDVI)
        composite_ds = dat_full_ds_top2NDVI_lowestVZA.where(~nan_inds_NDVI, dat_full_ds_top2NDVI_highestVZA)

    # del nan_inds_NDVI, dat_full_ds_top2NDVI_lowestVZA, dat_full_ds_top2NDVI_highestVZA
    # gc.collect() # manual garbage collection


    # clean up the dataset for saving
    # vars_to_drop = ['time', 'timeobs', 'Orbits', 'NDVI_rank', 'VZA_rank']
    # composite_ds = composite_ds.drop(vars_to_drop)
    composite_dayofyear = (composite_period * 16) + 1
    dat_date = dt.strptime(f"{composite_year}{composite_dayofyear:03d}", "%Y%j")
    composite_ds = composite_ds.expand_dims(time = [dat_date])

    composite_ds = composite_ds.assign(CCI=(composite_ds["sur_refl_b11"] - composite_ds["sur_refl_b01"])/
                            (composite_ds["sur_refl_b11"] + composite_ds["sur_refl_b01"]))


    # # Define attributes for all variables being saved in netcdf
    composite_ds.CCI.attrs['long_name'] = 'Chlorophyll Carotenoid Index'
    composite_ds.NDVI.attrs['long_name'] = 'Normalized Difference Vegetation Index'

    # include formulas
    composite_ds.CCI.attrs['formula'] = '(B11 - B1)/(B11 + B1)'
    composite_ds.NDVI.attrs['formula'] = '(B2 - B1)/(B2 + B1)'


    # define global attributes
    composite_ds.attrs['title'] = 'MODIS MOD09 VIs calculated from surface reflectance - 16-day composite using Constrained View angle - Maximum Value Composite (CV-MVC) algorithm'
    composite_ds.attrs['date_string'] = str(dat_date.date())
    composite_ds.attrs['Comment1'] = 'Original files downloaded from Google Earth Engine'
    composite_ds.attrs['DOI'] = 'https://doi.org/10.5067/MODIS/MOD09GA.061'
    composite_ds.attrs['Citation'] = 'Vermote, E., & Wolfe, R. (2021). MODIS/Terra Surface Reflectance Daily L2G Global 1km and 500m SIN Grid V061 [Data set]. NASA Land Processes Distributed Active Archive Center.'
    composite_ds.attrs['Processed_by'] = 'Lewis Kunik, lewis.kunik@utah.edu'
    composite_ds.attrs['Date_created'] = str(dt.today().date())

    # Finally, save to netcdf
    composite_ds.to_netcdf(path=outfile, format="NETCDF4")
    composite_ds.close()

    end_total = time.time()
    total_elapsed = timedelta(seconds=end_total-start_total)
    print(f'File saved. Total elapsed time = {td_min(total_elapsed)} mins, {td_sec(total_elapsed)} seconds')

if __name__ == "__main__":
    main()
