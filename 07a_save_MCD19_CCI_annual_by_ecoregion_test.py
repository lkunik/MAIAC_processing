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
mask_LC_high = True
mask_LC_low = False

pixel_presence_threshold = 5  # threshold for % of possible pixels that must be present in a composite period to include it in the analysis


EPA_ecoregion_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/EPA/ecoregions/'
EPA_ecoregion_L1_file = os.path.join(EPA_ecoregion_dir, 'L1', 'NA_CEC_Eco_Level1_EPSG4326_regrid.shp')
EPA_ecoregion_Eastern_file = os.path.join(EPA_ecoregion_dir, 'L3', 'Eastern_Forests.shp')
EPA_ecoregion_boreal_file = os.path.join(EPA_ecoregion_dir, 'L3', 'NAm_cec_eco_l3_Boreal_combined.shp')
EPA_ecoregion_WUS_file = os.path.join(EPA_ecoregion_dir, 'L3', 'NAm_cec_eco_l3_Temperate_ENF.shp')
EPA_ecoregion_US_Mex_file = os.path.join(EPA_ecoregion_dir, 'L2', 'na_cec_eco_l2_ENF_US_Mex.shp')


ESA_CCI_LC_mask_dir = "/uufs/chpc.utah.edu/common/home/lin-group23/ltk/ESA_CCI/LandCover/"
# ESA_CCI_CRO_mask_file = os.path.join(ESA_CCI_LC_mask_dir, "crop_with_mosaic_fraction_0.01deg_north_america.nc")
# ESA_CCI_DBF_closed_mask_file = os.path.join(ESA_CCI_LC_mask_dir, "broadleaved_deciduous_closed_forest_fraction_0.01deg_north_america.nc")
ESA_CCI_mixed_forest_mask_file = os.path.join(ESA_CCI_LC_mask_dir, "mixed_forest_fraction_0.01deg_north_america.nc")
ESA_CCI_ENF_openclosed_mask_file = os.path.join(ESA_CCI_LC_mask_dir, "needleleaf_evergreen_open_closed_forest_fraction_0.01deg_north_america.nc")
ESA_CCI_ENF_closed_mask_file = os.path.join(ESA_CCI_LC_mask_dir, "needleleaf_evergreen_closed_forest_fraction_0.01deg_north_america.nc")
ESA_CCI_Grass_mask_file = os.path.join(ESA_CCI_LC_mask_dir, "grassland_fraction_0.01deg_north_america.nc")
ESA_CCI_Shrub_mask_file = os.path.join(ESA_CCI_LC_mask_dir, "shrubland_fraction_0.01deg_north_america.nc")
# ESA_CCI_Wetland_mask_file = os.path.join(ESA_CCI_LC_mask_dir, "wetlands_fraction_0.01deg_north_america.nc")

LC_threshold_high = 0.8
LC_threshold_low = 0.2



########################################
# Define Global Variables and constants
#########################################
#%%


ecoregions_list = ['Boreal', 'Eastern', 'Western', 'Southwest US Mexico', 'GREAT PLAINS', 'NORTH AMERICAN DESERTS']

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


ecoregions_tiles_list={
    'Boreal': ['h10v02', 'h10v03', 'h11v02', 'h11v03', 'h12v02', 'h12v03', 'h12v04', 'h13v02', 'h13v03', 'h13v04', 'h14v03'],
    'Eastern': ['h11v04', 'h11v05', 'h12v04', 'h13v04','h10v05'], #['h09v05', 'h10v05', 'h10v06', 'h11v04', 'h11v05', 'h12v04', 'h12v05', 'h13v04'],
    'Western': ['h08v04', 'h08v05', 'h09v04', 'h09v05', 'h10v03', 'h10v04'],
    'Southwest US Mexico': ['h08v06', 'h08v05', 'h09v05'],
    'TUNDRA': ['h11v02', 'h12v02', 'h13v02', 'h14v02', 'h15v02', 'h14v03','h12v01', 'h13v01', 'h14v01', 'h15v01'],
    'TAIGA': ['h10v02', 'h11v02', 'h12v02', 'h11v03', 'h12v03', 'h13v03', 'h14v03'],
    'NORTHERN FORESTS': ['h11v03', 'h12v03', 'h13v03', 'h14v03', 'h11v04', 'h12v04', 'h13v04', 'h14v04'],
    'HUDSON PLAIN': ['h12v03', 'h13v03'],
    'NORTHWESTERN FORESTED MOUNTAINS': ['h11v02', 'h12v02', 'h10v03', 'h11v03', 'h08v04', 'h09v04', 'h10v04', 'h08v05', 'h09v05'],
    'MARINE WEST COAST FOREST': ['h10v02', 'h09v03', 'h10v03', 'h11v03', 'h09v04', 'h08v04','h08v05'],
    "EASTERN TEMPERATE FORESTS": ['h11v04', 'h12v04', 'h13v04', 'h10v05', 'h11v05', 'h12v05', 'h10v06'],
    "GREAT PLAINS": ['h11v03', 'h10v04', 'h11v04', 'h09v05', 'h10v05', 'h09v06'],
    "NORTH AMERICAN DESERTS": ['h08v04', 'h09v04', 'h10v04', 'h08v05', 'h09v05', 'h07v06', 'h08v06'],
    "MEDITERRANEAN CALIFORNIA": ['h08v04', 'h08v05'],
    "SOUTHERN SEMIARID HIGHLANDS": ['h08v05', 'h08v06'],
    "TEMPERATE SIERRAS": ['h08v05', 'h09v05', 'h08v06', 'h08v07', 'h09v07'],
    "TROPICAL WET FORESTS": ['h08v06', 'h09v06', 'h10v06', 'h08v07', 'h09v07'],
    "TROPICAL DRY FORESTS": ['h07v06', 'h08v06', 'h09v06', 'h08v07', 'h09v07']
}
#########################################
# Define Global Functions
#########################################

def dissolve_ecoregion_shapefile(ecoregions_gdf, buffer_dist = 0.001):
  # buffer_dist is in degrees, tune based on your gap size

    ecoregions_gdf['geometry'] = (
        ecoregions_gdf.geometry
        .buffer(buffer_dist)
        .buffer(-buffer_dist)
    )
    return ecoregions_gdf.dissolve()



#########################################
# Begin Main
#########################################

#%%

# directories
QC_option = 'CloudFree_LowAOD_ClearAdj'
MCD19_basedir = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/01d_rect/QCfilt_{QC_option}/'

plot_basedir = f'/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/plots/annual_ts/QCfilt_{QC_option}/'
Path(plot_basedir).mkdir(parents=True, exist_ok=True)

# mark start time to keep track of elapsed
start_time = time.time()

analysis_years = np.arange(2000, 2026)
tick_years = [2000, 2005, 2010, 2015, 2020, 2025]



# print(ecoregion_l2_gdf['NA_L2NAME'].unique())

ecoregion_dat_dict = {}


# Select sample ecoregion for testing
ecoregion = "Southwest US Mexico"

tiles_to_process = ecoregions_tiles_list[ecoregion]

avg_period = 'JJA'
# for ecoregion in ecoregions_l2_list:
print(f'Processing ecoregion: {ecoregion}...')

ecoregions_Eastern_gdf = gpd.read_file(EPA_ecoregion_Eastern_file)
ecoregions_boreal_gdf = gpd.read_file(EPA_ecoregion_boreal_file)
ecoregions_WUS_gdf = gpd.read_file(EPA_ecoregion_WUS_file)
ecoregions_US_Mex_gdf = gpd.read_file(EPA_ecoregion_US_Mex_file)



if ecoregion in ecoregions_l1_list:
    ecoregions_gdf = gpd.read_file(EPA_ecoregion_L1_file)
    # print(ecoregion_l2_gdf['NA_L2NAME'].unique())
    ecoregions_gdf_this_eco = ecoregions_gdf[ecoregions_gdf.NA_L1NAME == ecoregion]
    ecoregion_gdf = ecoregions_gdf_this_eco[ecoregions_gdf_this_eco['Shape_Area'] > 1e+9] # filter out small areas
    tiles_to_process = ecoregions_l1_tiles_list[ecoregion]
elif ecoregion == 'Boreal':
    ecoregions_gdf = ecoregions_boreal_gdf[ecoregions_boreal_gdf['Shape_Area'] > 1e+9]
    ecoregion_gdf = dissolve_ecoregion_shapefile(ecoregions_gdf)
elif ecoregion == 'Eastern':
    ecoregions_gdf = ecoregions_Eastern_gdf[ecoregions_Eastern_gdf['Shape_Area'] > 1e+9]
    ecoregion_gdf = dissolve_ecoregion_shapefile(ecoregions_gdf)
elif ecoregion == 'Western':
    ecoregions_gdf = ecoregions_WUS_gdf[ecoregions_WUS_gdf['Shape_Area'] > 1e+9]
    ecoregion_gdf = dissolve_ecoregion_shapefile(ecoregions_gdf)
elif ecoregion == 'Southwest_US_Mexico':
    ecoregions_gdf = ecoregions_US_Mex_gdf[ecoregions_US_Mex_gdf['Shape_Area'] > 1e+9]
    ecoregion_gdf = dissolve_ecoregion_shapefile(ecoregions_gdf)

ecoregion_gdf_geom = ecoregion_gdf.geometry


n_parts = len(list(ecoregion_gdf_geom.iloc[0].geoms)) if ecoregion_gdf_geom.iloc[0].geom_type == 'MultiPolygon' else 1
n_verts = sum(len(p.exterior.coords) + sum(len(i.coords) for i in p.interiors) 
              for p in (ecoregion_gdf_geom.iloc[0].geoms if n_parts > 1 else [ecoregion_gdf_geom.iloc[0]]))
if n_verts > 1e+6:
    print(f'vertices before simplify: {n_verts}, simplifying geometry...')
    ecoregion_gdf_geom = ecoregion_gdf_geom.simplify(0.005, preserve_topology=True)

    n_verts_simplified = sum(len(p.exterior.coords) + sum(len(i.coords) for i in p.interiors) 
                            for p in (ecoregion_gdf_geom.iloc[0].geoms 
                                        if ecoregion_gdf_geom.iloc[0].geom_type == 'MultiPolygon' 
                                        else [ecoregion_gdf_geom.iloc[0]]))
    print(f'vertices after simplify: {n_verts_simplified}')

#%%

print("Loading LC masks...")

# CRO_xr = xr.open_dataset(ESA_CCI_CRO_mask_file)
# DBF_closed_xr = xr.open_dataset(ESA_CCI_DBF_closed_mask_file)
ENF_openclosed_xr = xr.open_dataset(ESA_CCI_ENF_openclosed_mask_file, decode_coords='all')['fraction_ENF'].squeeze()
ENF_closed_xr = xr.open_dataset(ESA_CCI_ENF_closed_mask_file, decode_coords='all')['fraction_ENF'].squeeze()
Grass_xr = xr.open_dataset(ESA_CCI_Grass_mask_file, decode_coords='all')['fraction_GRA'].squeeze()
Shrub_xr = xr.open_dataset(ESA_CCI_Shrub_mask_file, decode_coords='all')['fraction_shrubland'].squeeze()
Mixed_forest_xr = xr.open_dataset(ESA_CCI_mixed_forest_mask_file, decode_coords='all')['fraction_mixed_forest'].squeeze()
# Wetland_xr = xr.open_dataset(ESA_CCI_Wetland_mask_file)

if ecoregion == "Boreal":
    LC_xr = ENF_openclosed_xr
elif ecoregion == "Eastern":
    LC_xr = Mixed_forest_xr
elif ecoregion == "Western":
    LC_xr = ENF_openclosed_xr
elif ecoregion == "Southwest US Mexico":
    LC_xr = ENF_openclosed_xr
elif ecoregion == "GREAT PLAINS":
    LC_xr = Grass_xr
elif ecoregion == "NORTH AMERICAN DESERTS":
    LC_xr = Shrub_xr


CCI_mean_arr = []
CCI_std_arr = []

NDVI_mean_arr = []
NDVI_std_arr = []
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


itile=0
MODIS_tile = tiles_to_process[itile]
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

    # Assume that both mask_LC_high and mask_LC_low cannot be True at the same time. If both are False, then no masking is applied.
    if mask_LC_high and mask_LC_low:
        raise ValueError("Both mask_LC_high and mask_LC_low cannot be True at the same time. Please set one of them to False.")

    # If masking for either high or low %Land Cover, perform filtering here
    if mask_LC_high | mask_LC_low:
        # Regrid Land Cover mask to this tile's grid if doesn't already exist in the dictionary
        LC_regrid = LC_xr.rio.write_crs("EPSG:4326").rio.reproject_match(MCD19_xr_tile, resampling=Resampling.bilinear, nodata=np.nan)

        if mask_LC_high:
            MCD19_xr_tile = MCD19_xr_tile.where(LC_regrid > LC_threshold_high)
        
        if mask_LC_low:
            MCD19_xr_tile = MCD19_xr_tile.where(LC_regrid < LC_threshold_low)

    MCD19_xr_clip = MCD19_xr_tile.rio.clip(ecoregion_gdf_geom.apply(mapping), drop=False)
    if MCD19_xr_clip['CCI'].isnull().all():
        print(f'    No valid data in clipped region for tile {MODIS_tile}, composite period {composite_period}')
        continue

    MCD19_xr_tile_ecoregion_mean = MCD19_xr_clip.mean(dim=['x', 'y'], skipna=True)
    MCD19_xr_tile_ecoregion_std = MCD19_xr_clip.std(dim=['x', 'y'], skipna=True)
    # MCD19_xr_spacetime_err = np.sqrt((MCD19_xr_composite_std['CCI']**2) + MCD19_xr_composite_mean['CCI_std']**2)
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


if mask_LC_high:
    output_file = os.path.join(output_dir, f'{ecoregion.replace(" ", "_")}_annual_dict_LCmask_only.pkl')
elif mask_LC_low:
    output_file = os.path.join(output_dir, f'{ecoregion.replace(" ", "_")}_annual_dict_LCmask_low.pkl')
else:
    output_file = os.path.join(output_dir, f'{ecoregion.replace(" ", "_")}_annual_dict.pkl')

with open(output_file, 'wb') as f:
    pickle.dump(ecoregion_dat_dict, f)

print(f'Saved ecoregion_annual_dict to {output_file}')

#%%

CCI_color = "#A1993D"
NDVI_color = "#006B30"

plt.rcParams.update({
'axes.labelsize': 20,
'axes.titlesize': 20,
'xtick.labelsize': 14,
'ytick.labelsize': 17,
'legend.fontsize': 13,
})


fig, ax = plt.subplots(figsize=(8, 3)) # establish the figure, axes
ax.grid(color = "gray", linestyle = "--", linewidth = 1, alpha = 0.5) # add grid lines
# in faint lines, plot the individual years of the data

ax.plot(CCI_std_xr_weighted.year, CCI_mean_xr_weighted, color=CCI_color, lw = 2.5)
# Plot shaded region for mean ± representative std
ax.fill_between(
    CCI_std_xr_weighted.year,
    np.array(CCI_mean_xr_weighted) - np.array(CCI_std_xr_weighted),
    np.array(CCI_mean_xr_weighted) + np.array(CCI_std_xr_weighted),
    color=CCI_color, alpha=0.5, label='±1 std'
)

# Define the axis, title labels
# set xticks to Month abbreviations
ax.set_xticks(ticks=CCI_std_xr_weighted.year.values[::2])
ax.set_xticklabels(CCI_std_xr_weighted.year.values[::2], rotation=45)
ax.set_xlim(CCI_std_xr_weighted.year[0]-0.5, CCI_std_xr_weighted.year[-1]+0.5)
# ax.set_ylim(PRB_min, PRB_max)

ax.set_ylabel('CCI')

plt.show()
# %%
