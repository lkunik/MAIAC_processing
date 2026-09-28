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
import gc

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
from rasterio.enums import Resampling

# shapefile/geospatial packages
import geopandas as gpd
from shapely.geometry import mapping

# time and date packages
import time
from datetime import datetime as dt  # date time library
from datetime import timedelta

# plotting packages
import pickle
from matplotlib import pyplot as plt  # primary plotting module

#%%
#########################################
# Define Global Filepaths
#########################################

# Select sample ecoregion for testing
# ecoregion = 'WESTERN CORDILLERA'
# ecoregion = 'BOREAL CORDILLERA'
# ecoregion = 'NORTHERN FORESTS'


mask_ENF = True


# directories
QC_option = 'CloudFree'
MCD19_basedir = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/01d_rect/QCfilt_{QC_option}/'

EPA_ecoregion_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/EPA/ecoregions/L3'
EPA_ecoregion_temperate_file = os.path.join(EPA_ecoregion_dir, 'NAm_cec_eco_l3_Temperate.shp')
EPA_ecoregion_boreal_file = os.path.join(EPA_ecoregion_dir, 'NAm_cec_eco_l3_Boreal_combined.shp')

plot_basedir = f'/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/plots/annual_ts/QCfilt_{QC_option}/'
Path(plot_basedir).mkdir(parents=True, exist_ok=True)


ENF_mask_file = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/ESA_CCI/LandCover/needleleaf_evergreen_open_closed_forest_fraction_0.01deg_north_america.nc'
ENF_threshold_high = 0.99
ENF_threshold_low = 0.2



########################################
# Define Global Variables and constants
#########################################
#%%

ecoregions_list = ['Temperate', 'Boreal']
ecoregions_tile_list={
    'Temperate': ['h10v03', 'h11v03', 
                  'h08v04', 'h09v04', 'h10v04', 'h11v04','h12v04', 'h13v04',
                  'h08v05', 'h09v05', 'h10v05', 'h11v05',  'h12v05', 
                  'h08v06', 'h09v06', 'h08v07'],
    'Boreal': ['h10v02', 'h11v02', 'h12v02', 'h13v02', 
               'h10v03',  'h11v03', 'h12v03', 'h13v03', 'h14v03']
}


#########################################
# Define Global Functions
#########################################



#########################################
# Begin Main
#########################################




# print(ecoregion_l2_gdf['NA_L2NAME'].unique())
analysis_years = np.arange(2000, 2026)
tick_years = [2000, 2005, 2010, 2015, 2020, 2025]

ecoregions_temperate_gdf = gpd.read_file(EPA_ecoregion_temperate_file)
ecoregions_boreal_gdf = gpd.read_file(EPA_ecoregion_boreal_file)

ecoregion_temperate_gdf = ecoregions_temperate_gdf.dissolve()
ecoregion_boreal_gdf = ecoregions_boreal_gdf.dissolve()

ecoregion_temperate_geom = ecoregion_temperate_gdf.geometry
ecoregion_boreal_geom = ecoregion_boreal_gdf.geometry

tiles_to_process_temperate = ecoregions_tile_list['Temperate']
tiles_to_process_boreal = ecoregions_tile_list['Boreal']


print("Loading ENF mask...")
ENF_xr = xr.open_dataset(ENF_mask_file, decode_coords='all')['fraction_ENF'].squeeze()


#%%
# mark start time to keep track of elapsed
start_time = time.time()

# ecoregion = 'Boreal'
# ecoregion_gdf_geom = ecoregion_boreal_geom
# tiles_to_process = tiles_to_process_boreal
ecoregion = 'Temperate'
ecoregion_gdf_geom = ecoregion_temperate_geom
tiles_to_process = tiles_to_process_temperate

ENF_regrids = {}
ecoregion_dat_dict = {}
analysis_years = np.arange(2000, 2026)

avg_period = 'JJA'
# for ecoregion in ecoregions_l2_list:
print(f'Processing ecoregion: {ecoregion}...')

ecoregion_dat_dict['CCI_mean'] = []
ecoregion_dat_dict['CCI_std'] = []
ecoregion_dat_dict['NDVI_mean'] = []
ecoregion_dat_dict['NDVI_std'] = []
# yyyy = analysis_years[0]
# for yyyy in analysis_years:
#     print(f'  Processing year: {yyyy}...')

CCI_mean_pd_arr = []
CCI_std_pd_arr = []
NDVI_mean_pd_arr = []
NDVI_std_pd_arr = []
npixels_pd_arr = []
Refl1_mean_pd_arr = []
Refl1_std_pd_arr = []
Refl2_mean_pd_arr = []
Refl2_std_pd_arr = []
Refl11_mean_pd_arr = []
Refl11_std_pd_arr = []
# MODIS_tile = tiles_to_process[0]
for itile, MODIS_tile in enumerate(tiles_to_process):
    print(f'  Processing tile: {MODIS_tile}...')
    MCD19_tile_dir = os.path.join(MCD19_basedir, MODIS_tile, 'annualized')
    #MCD19A1.h10v05.061_01d_annual_JJA_CV-MVC_QCfilt_CloudFree.nc
    MCD19_tile_file = os.path.join(MCD19_tile_dir, f'MCD19A1.{MODIS_tile}.061_01d_annual_{avg_period}_CV-MVC_QCfilt_{QC_option}.nc')

    if not os.path.exists(MCD19_tile_file):
        print(f'    File not found: {MCD19_tile_file}. Skipping this tile.')
        continue    
    MCD19_xr_tile = xr.open_dataset(MCD19_tile_file, decode_coords = 'all')
    MCD19_xr_tile.rio.write_crs("EPSG:4326", inplace=True)


    # Regrid ENF mask to this tile's grid if doesn't already exist in the dictionary
    if MODIS_tile not in ENF_regrids:
        print("  Regridding ENF mask to tile grid...")
        ENF_regrids[MODIS_tile] = ENF_xr.rio.write_crs("EPSG:4326").rio.reproject_match(MCD19_xr_tile, resampling=Resampling.bilinear, nodata=np.nan)
        print('  done regridding ENF mask.')
    ENF_regrid = ENF_regrids[MODIS_tile]

    
    if mask_ENF:
        print('  Applying ENF mask...')
        MCD19_xr_ENF = MCD19_xr_tile.where(ENF_regrid > ENF_threshold_high)
    else:
        print('  Applying non-ENF mask...')
        MCD19_xr_ENF = MCD19_xr_tile.where(ENF_regrid < ENF_threshold_low)

    print('  Clipping to ecoregion...')
    MCD19_xr_clip = MCD19_xr_ENF.rio.clip(ecoregion_gdf_geom.apply(mapping), drop=False)

    if MCD19_xr_clip["Sur_refl11"].isnull().all():
        print(f'    No valid ENF data in clipped region for tile {MODIS_tile}')
        continue

    MCD19_xr_clip = MCD19_xr_clip.assign(CCI=(MCD19_xr_clip["Sur_refl11"] - MCD19_xr_clip["Sur_refl1"])/
            (MCD19_xr_clip["Sur_refl11"] + MCD19_xr_clip["Sur_refl1"]))
    MCD19_xr_clip = MCD19_xr_clip.assign(NDMI=(MCD19_xr_clip["Sur_refl6"] - MCD19_xr_clip["Sur_refl1"])/
                (MCD19_xr_clip["Sur_refl6"] + MCD19_xr_clip["Sur_refl1"]))
    MCD19_xr_clip = MCD19_xr_clip.assign(NDVI=(MCD19_xr_clip["Sur_refl2"] - MCD19_xr_clip["Sur_refl1"])/
                (MCD19_xr_clip["Sur_refl2"] + MCD19_xr_clip["Sur_refl1"]))
    MCD19_xr_clip = MCD19_xr_clip.assign(NDSI=(MCD19_xr_clip["Sur_refl4"] - MCD19_xr_clip["Sur_refl6"])/
                (MCD19_xr_clip["Sur_refl4"] + MCD19_xr_clip["Sur_refl6"]))
    MCD19_xr_clip = MCD19_xr_clip.assign(NIRv=(MCD19_xr_clip["Sur_refl2"] - MCD19_xr_clip["Sur_refl1"])*MCD19_xr_clip["Sur_refl2"]/
                (MCD19_xr_clip["Sur_refl2"] + MCD19_xr_clip["Sur_refl1"]))



    print('  Calculating spatial mean for this tile...')
    MCD19_xr_tile_ecoregion_mean = MCD19_xr_clip.mean(dim=['x', 'y'], skipna=True)

    print('  Calculating spatial std for this tile...')
    MCD19_xr_tile_ecoregion_std = MCD19_xr_clip.std(dim=['x', 'y'], skipna=True)
    # MCD19_xr_spacetime_err = np.sqrt((MCD19_xr_composite_std['CCI']**2) + MCD19_xr_composite_mean['CCI_std']**2)

    print('  Counting valid pixels for this tile...')
    MCD19_xr_npixels = MCD19_xr_clip['CCI'].count(dim=['x', 'y'])

    # Convert to DataArrays with itile dimension for concatenation
    CCI_mean_tile = MCD19_xr_tile_ecoregion_mean['CCI'].expand_dims({'itile': [itile]})
    CCI_std_tile = MCD19_xr_tile_ecoregion_std['CCI'].expand_dims({'itile': [itile]})
    NDVI_mean_tile = MCD19_xr_tile_ecoregion_mean['NDVI'].expand_dims({'itile': [itile]})
    NDVI_std_tile = MCD19_xr_tile_ecoregion_std['NDVI'].expand_dims({'itile': [itile]})
    Refl1_mean_tile = MCD19_xr_tile_ecoregion_mean['Sur_refl1'].expand_dims({'itile': [itile]})
    Refl1_std_tile = MCD19_xr_tile_ecoregion_std['Sur_refl1'].expand_dims({'itile': [itile]})
    Refl2_mean_tile = MCD19_xr_tile_ecoregion_mean['Sur_refl2'].expand_dims({'itile': [itile]})
    Refl2_std_tile = MCD19_xr_tile_ecoregion_std['Sur_refl2'].expand_dims({'itile': [itile]})
    Refl11_mean_tile = MCD19_xr_tile_ecoregion_mean['Sur_refl11'].expand_dims({'itile': [itile]})
    Refl11_std_tile = MCD19_xr_tile_ecoregion_std['Sur_refl11'].expand_dims({'itile': [itile]})

    npixels_tile = MCD19_xr_npixels.expand_dims({'itile': [itile]})
    CCI_mean_pd_arr.append(CCI_mean_tile)
    CCI_std_pd_arr.append(CCI_std_tile)
    NDVI_mean_pd_arr.append(NDVI_mean_tile)
    NDVI_std_pd_arr.append(NDVI_std_tile)
    Refl1_mean_pd_arr.append(Refl1_mean_tile)
    Refl1_std_pd_arr.append(Refl1_std_tile)
    Refl2_mean_pd_arr.append(Refl2_mean_tile)
    Refl2_std_pd_arr.append(Refl2_std_tile)
    Refl11_mean_pd_arr.append(Refl11_mean_tile)
    Refl11_std_pd_arr.append(Refl11_std_tile)
    npixels_pd_arr.append(npixels_tile)

#%%

CCI_mean_xr_concat = xr.concat(CCI_mean_pd_arr, dim='itile')
CCI_std_xr_concat = xr.concat(CCI_std_pd_arr, dim='itile')
NDVI_mean_xr_concat = xr.concat(NDVI_mean_pd_arr, dim='itile')
NDVI_std_xr_concat = xr.concat(NDVI_std_pd_arr, dim='itile')
npixels_xr_concat = xr.concat(npixels_pd_arr, dim='itile')
Refl1_mean_xr_concat = xr.concat(Refl1_mean_pd_arr, dim='itile')
Refl1_std_xr_concat = xr.concat(Refl1_std_pd_arr, dim='itile')
Refl2_mean_xr_concat = xr.concat(Refl2_mean_pd_arr, dim='itile')
Refl2_std_xr_concat = xr.concat(Refl2_std_pd_arr, dim='itile')
Refl11_mean_xr_concat = xr.concat(Refl11_mean_pd_arr, dim='itile')
Refl11_std_xr_concat = xr.concat(Refl11_std_pd_arr, dim='itile')
all_pixels = npixels_xr_concat.sum(dim='itile')
weights_xr = npixels_xr_concat / all_pixels

CCI_mean_xr_weighted = (CCI_mean_xr_concat * weights_xr).sum(dim='itile')
CCI_std_xr_weighted = (CCI_std_xr_concat * weights_xr).sum(dim='itile')
NDVI_mean_xr_weighted = (NDVI_mean_xr_concat * weights_xr).sum(dim='itile')
NDVI_std_xr_weighted = (NDVI_std_xr_concat * weights_xr).sum(dim='itile')
Refl1_mean_xr_weighted = (Refl1_mean_xr_concat * weights_xr).sum(dim='itile')
Refl1_std_xr_weighted = (Refl1_std_xr_concat * weights_xr).sum(dim='itile')
Refl2_mean_xr_weighted = (Refl2_mean_xr_concat * weights_xr).sum(dim='itile')
Refl2_std_xr_weighted = (Refl2_std_xr_concat * weights_xr).sum(dim='itile')
Refl11_mean_xr_weighted = (Refl11_mean_xr_concat * weights_xr).sum(dim='itile')
Refl11_std_xr_weighted = (Refl11_std_xr_concat * weights_xr).sum(dim='itile')


ecoregion_dat_dict = {
    'CCI_mean': CCI_mean_xr_weighted.values,
    'CCI_std': CCI_std_xr_weighted.values,
    'NDVI_mean': NDVI_mean_xr_weighted.values,
    'NDVI_std': NDVI_std_xr_weighted.values,
    'years': CCI_mean_xr_weighted.year.values,
    'Refl1_mean': Refl1_mean_xr_weighted.values,
    'Refl1_std': Refl1_std_xr_weighted.values,
    'Refl2_mean': Refl2_mean_xr_weighted.values,
    'Refl2_std': Refl2_std_xr_weighted.values,
    'Refl11_mean': Refl11_mean_xr_weighted.values,
    'Refl11_std': Refl11_std_xr_weighted.values
}


# import pickle
output_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/data/summary_data'
Path(output_dir).mkdir(parents=True, exist_ok=True)

if mask_ENF:
    output_file = os.path.join(output_dir, f'{ecoregion.replace(" ", "_")}_annual_dict_ENF_only.pkl')
else:
    output_file = os.path.join(output_dir, f'{ecoregion.replace(" ", "_")}_annual_dict_nonENF.pkl')

with open(output_file, 'wb') as f:
    pickle.dump(ecoregion_dat_dict, f)

print(f'Saved ecoregion_annual_dict to {output_file}')

#%%
