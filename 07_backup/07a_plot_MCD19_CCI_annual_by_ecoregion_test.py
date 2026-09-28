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

# directories
QC_option = 'CloudFree'
MCD19_basedir = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/01d_rect/QCfilt_{QC_option}/'

EPA_ecoregion_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/EPA/ecoregions/'
EPA_ecoregion_L1_file = os.path.join(EPA_ecoregion_dir, 'L1', 'NA_CEC_Eco_Level1_EPSG4326_regrid.shp')
EPA_ecoregion_L2_file = os.path.join(EPA_ecoregion_dir, 'L2/NAm_shp_FOR_MAIAC_PROC', 'NA_CEC_Eco_Level2_withOther.shp')
EPA_ecoregion_L3_file = os.path.join(EPA_ecoregion_dir, 'L3/NAm_shp_FOR_MAIAC_PROC', 'NAm_cec_eco_l3_v2.shp')

plot_basedir = f'/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/plots/annual_ts/QCfilt_{QC_option}/'
Path(plot_basedir).mkdir(parents=True, exist_ok=True)

# eddy covariance data


########################################
# Define Global Variables and constants
#########################################
#%%


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

                      
all_tiles_list = sorted(set(tile for tiles in ecoregions_l2_tiles_list.values() for tile in tiles))
#########################################
# Define Global Functions
#########################################



#########################################
# Begin Main
#########################################

#%%
# mark start time to keep track of elapsed
start_time = time.time()

analysis_years = np.arange(2000, 2026)


ecoregion_l2_gdf = gpd.read_file(EPA_ecoregion_L2_file)

# print(ecoregion_l2_gdf['NA_L2NAME'].unique())

ecoregion_dat_dict = {}

ecoregion = ecoregions_l2_list[0]
avg_period = 'JJA'
# for ecoregion in ecoregions_l2_list:
print(f'Processing ecoregion: {ecoregion}...')
tiles_to_process = ecoregions_l2_tiles_list[ecoregion]

ecoregion_gdf = ecoregion_l2_gdf[ecoregion_l2_gdf.NA_L2NAME == ecoregion]
ecoregion_gdf_geom = ecoregion_gdf.geometry

CCI_mean_arr = []
CCI_std_arr = []
# yyyy = analysis_years[0]
# for yyyy in analysis_years:
#     print(f'  Processing year: {yyyy}...')

CCI_mean_pd_arr = []
CCI_std_pd_arr = []
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
    MCD19_xr_tile.rio.write_crs("EPSG:4326", inplace=True)

    MCD19_xr_clip = MCD19_xr_tile.rio.clip(ecoregion_gdf_geom.apply(mapping), drop=False)


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


    MCD19_xr_tile_ecoregion_mean = MCD19_xr_clip.mean(dim=['x', 'y'], skipna=True)
    MCD19_xr_tile_ecoregion_std = MCD19_xr_clip.std(dim=['x', 'y'], skipna=True)
    # MCD19_xr_spacetime_err = np.sqrt((MCD19_xr_composite_std['CCI']**2) + MCD19_xr_composite_mean['CCI_std']**2)
    MCD19_xr_npixels = MCD19_xr_clip['CCI'].count(dim=['x', 'y'])

    # Convert to DataArrays with itile dimension for concatenation
    CCI_mean_tile = MCD19_xr_tile_ecoregion_mean['CCI'].expand_dims({'itile': [itile]})
    CCI_std_tile = MCD19_xr_tile_ecoregion_std['CCI'].expand_dims({'itile': [itile]})
    npixels_tile = MCD19_xr_npixels.expand_dims({'itile': [itile]})
    Refl1_mean_tile = MCD19_xr_tile_ecoregion_mean['Sur_refl1'].expand_dims({'itile': [itile]})
    Refl1_std_tile = MCD19_xr_tile_ecoregion_std['Sur_refl1'].expand_dims({'itile': [itile]})
    Refl2_mean_tile = MCD19_xr_tile_ecoregion_mean['Sur_refl2'].expand_dims({'itile': [itile]})
    Refl2_std_tile = MCD19_xr_tile_ecoregion_std['Sur_refl2'].expand_dims({'itile': [itile]})
    Refl11_mean_tile = MCD19_xr_tile_ecoregion_mean['Sur_refl11'].expand_dims({'itile': [itile]})
    Refl11_std_tile = MCD19_xr_tile_ecoregion_std['Sur_refl11'].expand_dims({'itile': [itile]})


    CCI_mean_pd_arr.append(CCI_mean_tile)
    CCI_std_pd_arr.append(CCI_std_tile)
    npixels_pd_arr.append(npixels_tile)
    Refl1_mean_pd_arr.append(Refl1_mean_tile)
    Refl1_std_pd_arr.append(Refl1_std_tile)
    Refl2_mean_pd_arr.append(Refl2_mean_tile)
    Refl2_std_pd_arr.append(Refl2_std_tile)
    Refl11_mean_pd_arr.append(Refl11_mean_tile)
    Refl11_std_pd_arr.append(Refl11_std_tile)

#%%

CCI_mean_xr_concat = xr.concat(CCI_mean_pd_arr, dim='itile')
CCI_std_xr_concat = xr.concat(CCI_std_pd_arr, dim='itile')
npixels_xr_concat = xr.concat(npixels_pd_arr, dim='itile')
Refl1_mean_xr_concat = xr.concat(Refl1_mean_pd_arr, dim='itile')
Refl1_std_xr_concat = xr.concat(Refl1_std_pd_arr, dim='itile')
Refl2_mean_xr_concat = xr.concat(Refl2_mean_pd_arr, dim='itile')
Refl2_std_xr_concat = xr.concat(Refl2_std_pd_arr, dim='itile')
Refl11_mean_xr_concat = xr.concat(Refl11_mean_pd_arr, dim='itile')
Refl11_std_xr_concat = xr.concat(Refl11_std_pd_arr, dim='itile')

all_pixels = npixels_xr_concat.nansum(dim='itile')
weights_xr = npixels_xr_concat / all_pixels

CCI_mean_xr_weighted = (CCI_mean_xr_concat * weights_xr).sum(dim='itile')
CCI_std_xr_weighted = (CCI_std_xr_concat * weights_xr).sum(dim='itile')
Refl1_mean_xr_weighted = (Refl1_mean_xr_concat * weights_xr).sum(dim='itile')
Refl1_std_xr_weighted = (Refl1_std_xr_concat * weights_xr).sum(dim='itile')
Refl2_mean_xr_weighted = (Refl2_mean_xr_concat * weights_xr).sum(dim='itile')
Refl2_std_xr_weighted = (Refl2_std_xr_concat * weights_xr).sum(dim='itile')
Refl11_mean_xr_weighted = (Refl11_mean_xr_concat * weights_xr).sum(dim='itile')
Refl11_std_xr_weighted = (Refl11_std_xr_concat * weights_xr).sum(dim='itile')


ecoregion_dat_dict[ecoregion] = {
    'CCI_mean': CCI_mean_xr_weighted.values,
    'CCI_std': CCI_std_xr_weighted.values,
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

output_file = os.path.join(output_dir, f'{ecoregion.replace(" ", "_")}_annual_dict.pkl')
with open(output_file, 'wb') as f:
    pickle.dump(ecoregion_dat_dict, f)

print(f'Saved ecoregion_annual_dict to {output_file}')

#%%

plt.rcParams.update({
'axes.labelsize': 20,
'axes.titlesize': 20,
'xtick.labelsize': 14,
'ytick.labelsize': 17,
'legend.fontsize': 13,
})


fig, ax = plt.subplots(figsize=(7, 3)) # establish the figure, axes
ax.grid(color = "gray", linestyle = "--", linewidth = 1, alpha = 0.5) # add grid lines
# in faint lines, plot the individual years of the data

ax.plot(CCI_std_xr_weighted.year, CCI_mean_xr_weighted, color='black', lw = 2.5)
# Plot shaded region for mean ± representative std
ax.fill_between(
    CCI_std_xr_weighted.year,
    np.array(CCI_mean_xr_weighted) - np.array(CCI_std_xr_weighted),
    np.array(CCI_mean_xr_weighted) + np.array(CCI_std_xr_weighted),
    color='gray', alpha=0.3, label='±1 std'
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
