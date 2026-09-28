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

import numpy as np


#########################################
# Define Global Filepaths
#########################################




# ALL TILES BELOW

MODIS_tiles= [
# "h07v06", 
# "h08v03",
"h08v04",
"h08v05",
"h08v06",
# "h08v07",
# "h09v02",
# "h09v03" ,
"h09v04", # US-NR1
"h09v05", # ecoregions related to US-NR1
# "h09v06",
# "h09v07",
# "h09v08", 
"h10v02",   
"h10v03",
"h10v04", # ecoregions related to US-NR1
"h10v05",
"h10v06", # OSBS
# "h10v07",
# "h10v08",
"h11v02", # DEJU
"h11v03", # Ca-Obs
"h11v04",
"h11v05",
# "h11v06",
# "h11v07",
# "h11v08",
# "h12v01",
"h12v02",
"h12v03",
"h12v04",
"h12v05",
# "h13v01",
"h13v02",
"h13v03",
"h13v04",
# "h14v01",
"h14v02",
"h14v03",
# "h14v04",
# "h15v01",
# "h15v02",
# "h16v01",
]



#########################################
# Define Global functions
#########################################


#########################################
# Define Global Variables and constants
#########################################

start_year = 2000
end_year = 2025
file_years = np.arange(start_year, end_year+1)

#########################################
# Begin main
#########################################

#%%

QC_option = "CloudFree_LowAOD_ClearAdj" 




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 0 - historic average files from outlier-filtered data
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

dat_basedir = '/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/pre-fill_OutlierFilter/'

tiles_complete = []
tiles_to_reprocess = []
print(f'QC filtering Option: {QC_option}')
for MODIS_tile in MODIS_tiles:

    tile_files_missing = 0
    print("")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(f'tile {MODIS_tile}')
    dat_dir = os.path.join(dat_basedir, f'QCfilt_{QC_option}', MODIS_tile, 'hist_avg')

    for composite_nn in range(0, 23):

        # Skip processing if 'v01' is in MODIS_tile and composite_period is in [0, 1, 2, 19, 20, 21, 22]
        if 'v01' in MODIS_tile and composite_nn in [0, 1, 2, 19, 20, 21, 22]:
            continue

        # Skip processing if 'v02' is in MODIS_tile and composite_period is in [0, 1, 20, 21, 22]
        if 'v02' in MODIS_tile and composite_nn in [0, 1, 20, 21, 22]:
            continue
        
        #sample: MCD19A1.h12v03.061_1km_CV-MVC_QCfilt_CloudFree_hist_avg_20.nc
        filename = f'MCD19A1.{MODIS_tile}.061_1km_CV-MVC_QCfilt_{QC_option}_hist_avg_{composite_nn:02d}.nc'
        file_fullpath = f'{dat_dir}/{filename}'
        if not os.path.exists(file_fullpath):
            print(f'    File missing: {filename}')
            tile_files_missing += 1
    if tile_files_missing == 0:
        print(f'tile {MODIS_tile} - no files missing (all good)')        
        tiles_complete.append(MODIS_tile)
    else:
        print(f'tile {MODIS_tile} - {tile_files_missing} files missing')
        tiles_to_reprocess.append(MODIS_tile)
    
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("")

print(f'Step 0 DONE: Tiles with all hist-avg files done: {tiles_complete}')
print(f'             Tiles with missing hist-avg files : {tiles_to_reprocess}')



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1 - basic gap-filled files
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

dat_basedir = '/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/basic-fill/'

tiles_complete = []
tiles_to_reprocess = []
print(f'QC filtering Option: {QC_option}')
for MODIS_tile in MODIS_tiles:

    tile_files_missing = 0
    print("")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(f'tile {MODIS_tile}')
    dat_dir = os.path.join(dat_basedir, f'QCfilt_{QC_option}', MODIS_tile)
    for yyyy in file_years:
        for composite_nn in range(0, 23):

            # skip files before March 2000
            if ((yyyy == 2000) and np.isin(composite_nn, [0, 1, 2])):
                continue

            # Skip processing if 'v01' is in MODIS_tile and composite_period is in [0, 1, 2, 19, 20, 21, 22]
            if 'v01' in MODIS_tile and composite_nn in [0, 1, 2, 19, 20, 21, 22]:
                continue

            # Skip processing if 'v02' is in MODIS_tile and composite_period is in [0, 1, 20, 21, 22]
            if 'v02' in MODIS_tile and composite_nn in [0, 1, 20, 21, 22]:
                continue
            
            #sample: MCD19A1.h12v02.061_1km_16day_CV-MVC_basic-fill_QCfilt_CloudFree_LowAOD_ClearAdj_2013-03.nc
            filename = f'MCD19A1.{MODIS_tile}.061_1km_16day_CV-MVC_basic-fill_QCfilt_{QC_option}_{yyyy}-{composite_nn:02d}.nc'

            file_fullpath = f'{dat_dir}/{filename}'
            if not os.path.exists(file_fullpath):
                print(f'    File missing: {filename}')
                tile_files_missing += 1
    if tile_files_missing == 0:
        print(f'tile {MODIS_tile} - no files missing (all good)') 
        tiles_complete.append(MODIS_tile)       
    else:
        print(f'tile {MODIS_tile} - {tile_files_missing} files missing')
        tiles_to_reprocess.append(MODIS_tile)
    
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("")

print(f'Step 1 DONE: Tiles with all gap-filled files done: {tiles_complete}')
print(f'             Tiles with missing gap-filled files : {tiles_to_reprocess}')




# #%%
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # 6 - historic average gap-filled files
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# dat_basedir = '/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/lm-fill/'

# tiles_complete = []
# tiles_to_reprocess = []
# print(f'QC filtering Option: {QC_option}')
# for MODIS_tile in MODIS_tiles:

#     tile_files_missing = 0
#     print("")
#     print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#     print(f'tile {MODIS_tile}')
#     dat_dir = os.path.join(dat_basedir, f'QCfilt_{QC_option}', MODIS_tile, 'hist_avg')

#     for composite_nn in range(0, 23):

#         # Skip processing if 'v01' is in MODIS_tile and composite_period is in [0, 1, 2, 19, 20, 21, 22]
#         if 'v01' in MODIS_tile and composite_nn in [0, 1, 2, 19, 20, 21, 22]:
#             continue

#         # Skip processing if 'v02' is in MODIS_tile and composite_period is in [0, 1, 20, 21, 22]
#         if 'v02' in MODIS_tile and composite_nn in [0, 1, 20, 21, 22]:
#             continue
        
#         #sample: MCD19A1.h12v04.061_1km_CV-MVC_lmFill_QCfilt_CloudFree_hist_avg_12.nc
#         filename = f'MCD19A1.{MODIS_tile}.061_1km_CV-MVC_lmFill_QCfilt_{QC_option}_hist_avg_{composite_nn:02d}.nc'
#         file_fullpath = f'{dat_dir}/{filename}'
#         if not os.path.exists(file_fullpath):
#             print(f'    File missing: {filename}')
#             tile_files_missing += 1
#     if tile_files_missing == 0:
#         print(f'tile {MODIS_tile} - no files missing (all good)') 
#         tiles_complete.append(MODIS_tile)       
#     else:
#         print(f'tile {MODIS_tile} - {tile_files_missing} files missing')
#         tiles_to_reprocess.append(MODIS_tile)
    
#     print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#     print("")

# print(f'Step 6 DONE: Tiles with all gap-fill hist-avg files done: {tiles_complete}')
# print(f'             Tiles with missing gap-fill hist-avg files : {tiles_to_reprocess}')
