#!/usr/bin/env python3
"""
Read weekly GPP data from FLUXNET sites.
Extracts GPP_NT_VUT_50 from *FLUXNET_FULLSET_WW*.csv and *FLUXNET_FLUXMET_WW*.csv files.
"""
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

QC_option = 'CloudFree_LowAOD_ClearAdj'

# List MCD19 tiles
MCD19_basedir = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/01d_rect/QCfilt_{QC_option}'
MCD19_tiles = [d.name for d in Path(MCD19_basedir).iterdir() if d.is_dir()]

EPA_ecoregion_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/EPA/ecoregions/'
EPA_ecoregion_L1_file = os.path.join(EPA_ecoregion_dir, 'L1', 'NA_CEC_Eco_Level1_EPSG4326_regrid.shp')
EPA_ecoregion_L2_file = os.path.join(EPA_ecoregion_dir, 'L2/NAm_shp_FOR_MAIAC_PROC', 'NA_CEC_Eco_Level2_withOther.shp')
EPA_ecoregion_L3_file = os.path.join(EPA_ecoregion_dir, 'L3/NAm_shp_FOR_MAIAC_PROC', 'NAm_cec_eco_l3_v2.shp')
EPA_ecoregion_Eastern_file = os.path.join(EPA_ecoregion_dir, 'L3', 'Eastern_Forests.shp')
EPA_ecoregion_boreal_file = os.path.join(EPA_ecoregion_dir, 'L3', 'NAm_cec_eco_l3_Boreal_combined.shp')
EPA_ecoregion_WUS_file = os.path.join(EPA_ecoregion_dir, 'L3', 'NAm_cec_eco_l3_Temperate_ENF.shp')
EPA_ecoregion_US_Mex_file = os.path.join(EPA_ecoregion_dir, 'L2', 'na_cec_eco_l2_ENF_US_Mex.shp')

min_valid_threshold = 0.01

ecoregions_l1_list = ['TUNDRA',
                      'TAIGA', 
                      'NORTHERN FORESTS',
                      'HUDSON PLAIN',
                      'NORTHWESTERN FORESTED MOUNTAINS',
                      'MARINE WEST COAST FOREST', 
                      "EASTERN TEMPERATE FORESTS", 
                      "GREAT PLAINS", 
                      "NORTH AMERICAN DESERTS",
                      "MEDITERRANEAN CALIFORNIA",
                      "SOUTHERN SEMIARID HIGHLANDS",
                      "TEMPERATE SIERRAS",
                      "TROPICAL WET FORESTS",
                      "TROPICAL DRY FORESTS"]

ecoregions_l2_list = ['SOUTH CENTRAL SEMIARID PRAIRIES', 
                      'WESTERN CORDILLERA', 
                      'BOREAL CORDILLERA', 
                      "SOFTWOOD SHIELD", 
                      "SOUTHEASTERN USA PLAINS", 
                      "WARM DESERTS"]

ecoregions_to_exclude = ['WATER']

####################################
### Define Global Functions
####################################

def dissolve_ecoregion_shapefile(ecoregions_gdf, buffer_dist = 0.001):
  # buffer_dist is in degrees, tune based on your gap size

    ecoregions_gdf['geometry'] = (
        ecoregions_gdf.geometry
        .buffer(buffer_dist)
        .buffer(-buffer_dist)
    )
    return ecoregions_gdf.dissolve()


####################################
### Begin Main
####################################


ecoregions_L1_gdf = gpd.read_file(EPA_ecoregion_L1_file)
ecoregions_L2_gdf = gpd.read_file(EPA_ecoregion_L2_file)
ecoregions_Eastern_gdf = gpd.read_file(EPA_ecoregion_Eastern_file)
ecoregions_boreal_gdf = gpd.read_file(EPA_ecoregion_boreal_file)
ecoregions_WUS_gdf = gpd.read_file(EPA_ecoregion_WUS_file)
ecoregions_US_Mex_gdf = gpd.read_file(EPA_ecoregion_US_Mex_file)

ecoregion_Eastern_gdf = dissolve_ecoregion_shapefile(ecoregions_Eastern_gdf)
ecoregion_boreal_gdf = dissolve_ecoregion_shapefile(ecoregions_boreal_gdf)
ecoregion_WUS_gdf = dissolve_ecoregion_shapefile(ecoregions_WUS_gdf)
ecoregion_US_Mex_gdf = dissolve_ecoregion_shapefile(ecoregions_US_Mex_gdf)


ecoregion_Eastern_geom = ecoregion_Eastern_gdf.geometry
ecoregion_boreal_geom = ecoregion_boreal_gdf.geometry
ecoregion_WUS_geom = ecoregion_WUS_gdf.geometry
ecoregion_US_Mex_geom = ecoregion_US_Mex_gdf.geometry

# ecoregions_L3_gdf = gpd.read_file(EPA_ecoregion_L3_file)
print("LEVEL 1 ECOREGIONS:")
ecoregion_L1_names = ecoregions_L1_gdf['NA_L1NAME'].unique()
print(ecoregion_L1_names)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n\n")
print("LEVEL 2 ECOREGIONS:")
ecoregion_L2_names = ecoregions_L2_gdf['NA_L2NAME'].unique()
print(ecoregion_L2_names)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n\n")
# print("LEVEL 3 ECOREGIONS:")
# print(ecoregions_L3_gdf['NA_L3NAME'].unique())


ecoregions_L1_tiles_dict = {}
ecoregions_L2_tiles_dict = {}
ecoregions_Eastern_tiles_arr = []
ecoregions_boreal_tiles_arr = []
ecoregions_WUS_tiles_arr = []
ecoregions_US_Mex_tiles_arr = []

# tile = MCD19_tiles[0]
for tile in MCD19_tiles:

    print(f'Loading MCD19 tile: {tile}')
    tile_dir = os.path.join(MCD19_basedir, tile, "annualized")

    try:
        MCD19_tile_file = glob.glob(os.path.join(tile_dir, 'MCD19A1*_annual_JJA_*.nc'))[0]
    except IndexError:
        print(f"No annual JJA file found for tile {tile}")
        continue


    # Load the MCD19 tile as an xarray dataset and count number of valid values
    MCD19_xr_tile = xr.open_dataset(MCD19_tile_file, decode_coords = 'all')
    MCD19_xr_tile.rio.write_crs("EPSG:4326", inplace=True)
    MCD19_xr_tile_num_valid = int(MCD19_xr_tile['CCI'].notnull().sum().compute())
    if MCD19_xr_tile_num_valid == 0:
        print(f"No valid values for tile {tile}, skipping.")
        continue
    print(f"{tile}: {MCD19_xr_tile_num_valid} valid values")


    # # Check Eastern Temperate Forest regions
    # print("Checking Eastern Temperate Forest region")

    # MCD19_xr_clip = MCD19_xr_tile.rio.clip(ecoregion_Eastern_geom.apply(mapping), drop=False)
    # MCD19_xr_ecoregion_num_valid = int(MCD19_xr_clip['CCI'].notnull().sum().compute())
    # ecoregion_portion_valid = MCD19_xr_ecoregion_num_valid / MCD19_xr_tile_num_valid
    # print(f"{tile}: {MCD19_xr_ecoregion_num_valid} / {MCD19_xr_tile_num_valid} valid values ({ecoregion_portion_valid:.2%})")   
    # if ecoregion_portion_valid >= min_valid_threshold:
    #     ecoregions_Eastern_tiles_arr.append(tile)

    # # Clean up to free memory
    # del MCD19_xr_clip
    # gc.collect()

    # # Check Boreal Forest regions
    # print("Checking Boreal Forest region")

    # MCD19_xr_clip = MCD19_xr_tile.rio.clip(ecoregion_boreal_geom.apply(mapping), drop=False)
    # MCD19_xr_ecoregion_num_valid = int(MCD19_xr_clip['CCI'].notnull().sum().compute())
    # ecoregion_portion_valid = MCD19_xr_ecoregion_num_valid / MCD19_xr_tile_num_valid
    # print(f"{tile}: {MCD19_xr_ecoregion_num_valid} / {MCD19_xr_tile_num_valid} valid values ({ecoregion_portion_valid:.2%})")   
    # if ecoregion_portion_valid >= min_valid_threshold:
    #     ecoregions_boreal_tiles_arr.append(tile)

    # # Clean up to free memory
    # del MCD19_xr_clip
    # gc.collect()


    # Check WUS ENF regions
    print("Checking WUS ENF region")

    MCD19_xr_clip = MCD19_xr_tile.rio.clip(ecoregion_WUS_geom.apply(mapping), drop=False)
    MCD19_xr_ecoregion_num_valid = int(MCD19_xr_clip['CCI'].notnull().sum().compute())
    ecoregion_portion_valid = MCD19_xr_ecoregion_num_valid / MCD19_xr_tile_num_valid
    print(f"{tile}: {MCD19_xr_ecoregion_num_valid} / {MCD19_xr_tile_num_valid} valid values ({ecoregion_portion_valid:.2%})")   
    if ecoregion_portion_valid >= min_valid_threshold:
        ecoregions_WUS_tiles_arr.append(tile)

    # Clean up to free memory
    del MCD19_xr_clip
    gc.collect()


    # Check Southwest US/Mexico ENF regions
    print("Checking Southwest US/Mexico ENF region")

    MCD19_xr_clip = MCD19_xr_tile.rio.clip(ecoregion_US_Mex_geom.apply(mapping), drop=False)
    MCD19_xr_ecoregion_num_valid = int(MCD19_xr_clip['CCI'].notnull().sum().compute())
    ecoregion_portion_valid = MCD19_xr_ecoregion_num_valid / MCD19_xr_tile_num_valid
    print(f"{tile}: {MCD19_xr_ecoregion_num_valid} / {MCD19_xr_tile_num_valid} valid values ({ecoregion_portion_valid:.2%})")   
    if ecoregion_portion_valid >= min_valid_threshold:
        ecoregions_US_Mex_tiles_arr.append(tile)

    # Clean up to free memory
    del MCD19_xr_clip
    gc.collect()

    # # Loop through all the L1 ecoregions
    # print("Checking L1 ecoregions:")
    # ecoregion_name = ecoregion_L1_names[0]
    # for ecoregion_name in ecoregion_L1_names:

    #     if ecoregion_name in ecoregions_to_exclude:
    #         continue

    #     print(f"L1 ecoregion: {ecoregion_name}")
    #     ecoregion_gdf = ecoregions_L1_gdf[ecoregions_L1_gdf['NA_L1NAME'] == ecoregion_name]
    #     ecoregion_gdf_geom = ecoregion_gdf.geometry

    #     MCD19_xr_clip = MCD19_xr_tile.rio.clip(ecoregion_gdf_geom.apply(mapping), drop=False)
    #     MCD19_xr_ecoregion_num_valid = int(MCD19_xr_clip['CCI'].notnull().sum().compute())
    #     ecoregion_portion_valid = MCD19_xr_ecoregion_num_valid / MCD19_xr_tile_num_valid
    #     print(f"{tile}: {MCD19_xr_ecoregion_num_valid} / {MCD19_xr_tile_num_valid} valid values ({ecoregion_portion_valid:.2%})")

    #     if ecoregion_portion_valid >= min_valid_threshold:
    #         if ecoregion_name not in ecoregions_L1_tiles_dict:
    #             ecoregions_L1_tiles_dict[ecoregion_name] = []
    #         ecoregions_L1_tiles_dict[ecoregion_name].append(tile)
        
    #     # Clean up to free memory
    #     del MCD19_xr_clip
    #     gc.collect()

    # # Loop through all the L2 ecoregions
    # print("Checking L2 ecoregions:")
    # ecoregion_name = ecoregion_L2_names[0]
    # for ecoregion_name in ecoregion_L2_names:

    #     if ecoregion_name in ecoregions_to_exclude:
    #         continue

    #     print(f"L2 ecoregion: {ecoregion_name}")
    #     ecoregion_gdf = ecoregions_L2_gdf[ecoregions_L2_gdf['NA_L2NAME'] == ecoregion_name]
    #     ecoregion_gdf_geom = ecoregion_gdf.geometry

    #     MCD19_xr_clip = MCD19_xr_tile.rio.clip(ecoregion_gdf_geom.apply(mapping), drop=False)
    #     MCD19_xr_ecoregion_num_valid = int(MCD19_xr_clip['CCI'].notnull().sum().compute())
    #     ecoregion_portion_valid = MCD19_xr_ecoregion_num_valid / MCD19_xr_tile_num_valid
    #     print(f"{tile}: {MCD19_xr_ecoregion_num_valid} / {MCD19_xr_tile_num_valid} valid values ({ecoregion_portion_valid:.2%})")

    #     if ecoregion_portion_valid >= min_valid_threshold:
    #         if ecoregion_name not in ecoregions_L2_tiles_dict:
    #             ecoregions_L2_tiles_dict[ecoregion_name] = []
    #         ecoregions_L2_tiles_dict[ecoregion_name].append(tile)

    #     # Clean up to free memory
    #     del MCD19_xr_clip
    #     gc.collect()

    MCD19_xr_tile.close()
    del MCD19_xr_tile
    gc.collect()
    

print("Custom ecoregions:")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(f"Boreal Forest: {ecoregions_boreal_tiles_arr}")
print(f"Eastern Forest: {ecoregions_Eastern_tiles_arr}")
print(f"WUS ENF: {ecoregions_WUS_tiles_arr}")
print(f"Southwest US/Mexico ENF: {ecoregions_US_Mex_tiles_arr}")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n\n")



print("L1 ecoregions:")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
for ecoregions_L1_tiles_dict_ecoregion in ecoregions_L1_tiles_dict.items():
    ecoregion_name, tiles_list = ecoregions_L1_tiles_dict_ecoregion
    print(f"{ecoregion_name}: {tiles_list}")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n\n")


print("L2 ecoregions:")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
for ecoregions_L2_tiles_dict_ecoregion in ecoregions_L2_tiles_dict.items():
    ecoregion_name, tiles_list = ecoregions_L2_tiles_dict_ecoregion
    print(f"{ecoregion_name}: {tiles_list}")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n\n")


#####
# Custom ecoregions:
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Boreal Forest: ['h10v02', 'h10v03', 'h11v02', 'h11v03', 'h12v02', 'h12v03', 'h12v04', 'h13v02', 'h13v03', 'h13v04', 'h14v03']
# Eastern Forest: ['h09v05', 'h10v05', 'h10v06', 'h11v04', 'h11v05', 'h12v04', 'h12v05', 'h13v04']
# WUS ENF: ['h08v04', 'h08v05', 'h09v04', 'h09v05', 'h10v03', 'h10v04']
# Southwest US/Mexico ENF: ['h08v06', 'h08v05', 'h09v05']
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#
#
# L1 ecoregions:
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# NORTH AMERICAN DESERTS: ['h08v06', 'h08v04', 'h08v05', 'h09v04', 'h09v05', 'h10v03', 'h10v04']
# SOUTHERN SEMIARID HIGHLANDS: ['h08v06', 'h08v05']
# TEMPERATE SIERRAS: ['h08v06', 'h08v05', 'h09v05']
# TROPICAL DRY FORESTS: ['h08v06']
# TROPICAL WET FORESTS: ['h08v06', 'h10v06']
# GREAT PLAINS: ['h08v06', 'h09v05', 'h10v03', 'h10v04', 'h10v05', 'h11v03', 'h11v04']
# MEDITERRANEAN CALIFORNIA: ['h08v04', 'h08v05']
# NORTHWESTERN FORESTED MOUNTAINS: ['h08v04', 'h08v05', 'h09v04', 'h09v05', 'h10v02', 'h10v03', 'h10v04', 'h11v02', 'h11v03', 'h12v02']
# MARINE WEST COAST FOREST: ['h08v04', 'h09v04', 'h10v02', 'h10v03', 'h10v04', 'h11v02']
# EASTERN TEMPERATE FORESTS: ['h09v05', 'h10v05', 'h10v06', 'h11v04', 'h11v05', 'h12v04', 'h12v05', 'h13v04']
# TUNDRA: ['h10v02', 'h10v03', 'h11v02', 'h12v02', 'h13v02', 'h13v03', 'h14v02', 'h14v03']
# TAIGA: ['h10v02', 'h11v02', 'h11v03', 'h12v02', 'h12v03', 'h13v02', 'h13v03', 'h14v03']
# NORTHERN FORESTS: ['h10v03', 'h11v03', 'h11v04', 'h12v03', 'h12v04', 'h13v03', 'h13v04', 'h14v03']
# HUDSON PLAIN: ['h12v03', 'h13v03']
# ARCTIC CORDILLERA: ['h14v03']
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

