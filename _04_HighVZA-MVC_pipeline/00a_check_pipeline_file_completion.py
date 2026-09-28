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

QC_option = "CloudFree_LowAOD_ClearAdj" 

# Compare cloud masks:
# 1: AllCloud_AllAOD_AllAdj
# 2: LooseCloud_AllAOD_AllAdj
# 3: CloudFree_AllAOD_AllAdj

# Compare adjacency masks:
# 3: CloudFree_AllAOD_AllAdj
# 4: CloudFree_AllAOD_NotSurround
# 5: CloudFree_AllAOD_ClearAdj


# ALL TILES BELOW
MODIS_tiles = [	
# "h07v06",
# "h08v03",
"h08v04",
"h08v05",
"h08v06",
# "h08v07",
# "h09v02",
# "h09v03",
"h09v04",
"h09v05",
# "h09v06",
# "h09v07",
# "h09v08",
"h10v02",
"h10v03",
"h10v04",
"h10v05",
"h10v06",
# "h10v07",
# "h10v08",
"h11v02",
"h11v03",
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
# "h16v01"
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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1 - HighVZA-MVC pre-fill
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

dat_basedir = '/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/HighVZA-MVC/1km_curv/pre-fill/'

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
            
            #sample: MCD19A1.h12v02.061_1km_16day_CV-MVC_QCfilt_CloudFree_LowAOD_ClearAdj_2025-05.nc
            filename = f'MCD19A1.{MODIS_tile}.061_1km_16day_CV-MVC_QCfilt_{QC_option}_{yyyy}-{composite_nn:02d}.nc'

            file_fullpath = f'{dat_dir}/{filename}'
            if not os.path.exists(file_fullpath):
                print(f'    HighVZA-MVC File missing: {filename}')
                tile_files_missing += 1
    if tile_files_missing == 0:
        print(f'tile {MODIS_tile} - no HighVZA-MVC files missing (all good)') 
        tiles_complete.append(MODIS_tile)       
    else:
        print(f'tile {MODIS_tile} - {tile_files_missing} HighVZA-MVC files missing')
        tiles_to_reprocess.append(MODIS_tile)
    
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("")

print(f'Step 1 DONE: Tiles with all HighVZA-MVC files done: {tiles_complete}')
print(f'             Tiles with missing HighVZA-MVC files : {tiles_to_reprocess}')

# %%


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2 - historic average files
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

dat_basedir = '/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/HighVZA-MVC/1km_curv/pre-fill/'

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
        
        #sample: MCD19A1.h12v03.061_1km_HighVZA-MVC_QCfilt_CloudFree_hist_avg_20.nc
        filename = f'MCD19A1.{MODIS_tile}.061_1km_HighVZA-MVC_QCfilt_{QC_option}_hist_avg_{composite_nn:02d}.nc'
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

print(f'Step 2 DONE: Tiles with all hist-avg files done: {tiles_complete}')
print(f'             Tiles with missing hist-avg files : {tiles_to_reprocess}')


#%%

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2a - outlier-filtered pre-fill files (output from historical average step)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

dat_basedir = '/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/HighVZA-MVC/1km_curv/pre-fill_OutlierFilter/'

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
            
            #sample: MCD19A1.h09v04.061_1km_16day_HighVZA-MVC_QCfilt_CloudFree_LowAOD_ClearAdj_OutlierFilter_2025-16.nc
            filename = f'MCD19A1.{MODIS_tile}.061_1km_16day_HighVZA-MVC_QCfilt_{QC_option}_OutlierFilter_{yyyy}-{composite_nn:02d}.nc'

            file_fullpath = f'{dat_dir}/{filename}'
            if not os.path.exists(file_fullpath):
                print(f'    HighVZA-MVC Outlier-filtered File missing: {filename}')
                tile_files_missing += 1
    if tile_files_missing == 0:
        print(f'tile {MODIS_tile} - no HighVZA-MVC outlier-filtered files missing (all good)') 
        tiles_complete.append(MODIS_tile)       
    else:
        print(f'tile {MODIS_tile} - {tile_files_missing} HighVZA-MVC outlier-filtered files missing')
        tiles_to_reprocess.append(MODIS_tile)
    
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("")

print(f'Step 2a DONE: Tiles with all HighVZA-MVC outlier-filtered files done: {tiles_complete}')
print(f'             Tiles with missing HighVZA-MVC outlier-filtered files : {tiles_to_reprocess}')


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 3 - Scaling factors
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

dat_basedir = '/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/HighVZA-MVC/1km_curv/gapfill_scale/'
dat_dir = os.path.join(dat_basedir, f'QCfilt_{QC_option}')
tiles_complete = []
tiles_to_reprocess = []
print(f'QC filtering Option: {QC_option}')
for MODIS_tile in MODIS_tiles:

    tile_files_missing = 0
    print("")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(f'tile {MODIS_tile}')

    for composite_nn in range(0, 23):


        # Skip processing if 'v01' is in MODIS_tile and composite_period is in [0, 1, 2, 19, 20, 21, 22]
        if 'v01' in MODIS_tile and composite_nn in [0, 1, 2, 19, 20, 21, 22]:
            continue

        # Skip processing if 'v02' is in MODIS_tile and composite_period is in [0, 1, 20, 21, 22]
        if 'v02' in MODIS_tile and composite_nn in [0, 1, 20, 21, 22]:
            continue

        years_with_valid_scales = 0
        for yyyy in file_years:
            # skip files before March 2000
            if ((yyyy == 2000) and np.isin(composite_nn, [0, 1, 2])):
                continue

            #sample: h11v03_scale_factors_TAIGA_PLAIN_2001-04.csv
            filename_re = f'{MODIS_tile}_scale_factors_*_{yyyy}-{composite_nn:02d}.csv'
            all_files = glob.glob(f'{dat_dir}/{filename_re}')
            if len(all_files) > 0:
                # Not necessarily an issue if one year has all missing data
                years_with_valid_scales += 1
        if years_with_valid_scales == 0:
            print(f'    Warning: No years with valid scaling factor files found for tile {MODIS_tile} composite period {composite_nn:02d}')
            tile_files_missing += 1            
    if tile_files_missing == 0:
        print(f'tile {MODIS_tile} - no files missing (all good)')        
        tiles_complete.append(MODIS_tile)
    else:
        print(f'tile {MODIS_tile} - {tile_files_missing} composite periods with 0 ecoregion scaling factor files')
        tiles_to_reprocess.append(MODIS_tile)
    
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("")

print(f'Step 3 DONE: Tiles with all scaling factor files done: {tiles_complete}')
print(f'             Tiles with missing scaling factor files : {tiles_to_reprocess}')
# %%


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 4 - gap-filled files
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

dat_basedir = '/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/HighVZA-MVC/1km_curv/lm-fill/'

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
            
            #sample: MCD19A1.h13v03.061_1km_16day_HighVZA-MVC_lmFill_QCfilt_CloudFree_LowAOD_ClearAdj_2025-15.nc
            filename = f'MCD19A1.{MODIS_tile}.061_1km_16day_HighVZA-MVC_lmFill_QCfilt_{QC_option}_{yyyy}-{composite_nn:02d}.nc'

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

print(f'Step 4 DONE: Tiles with all gap-filled files done: {tiles_complete}')
print(f'             Tiles with missing gap-filled files : {tiles_to_reprocess}')



# #%%
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # 5 - regridded files
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# dat_basedir = '/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/HighVZA-MVC/01d_rect/'

# tiles_complete = []
# tiles_to_reprocess = []
# print(f'QC filtering Option: {QC_option}')
# for MODIS_tile in MODIS_tiles:

#     tile_files_missing = 0
#     print("")
#     print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#     print(f'tile {MODIS_tile}')
#     dat_dir = os.path.join(dat_basedir, f'QCfilt_{QC_option}', MODIS_tile)
#     for yyyy in file_years:
#         for composite_nn in range(0, 23):

#             # skip files before March 2000
#             if ((yyyy == 2000) and np.isin(composite_nn, [0, 1, 2])):
#                 continue

#             # Skip processing if 'v01' is in MODIS_tile and composite_period is in [0, 1, 2, 19, 20, 21, 22]
#             if 'v01' in MODIS_tile and composite_nn in [0, 1, 2, 19, 20, 21, 22]:
#                 continue

#             # Skip processing if 'v02' is in MODIS_tile and composite_period is in [0, 1, 20, 21, 22]
#             if 'v02' in MODIS_tile and composite_nn in [0, 1, 20, 21, 22]:
#                 continue
            
#             #sample: MCD19A1.h09v04.061_01d_16day_HighVZA-MVC_QCfilt_CloudFree_2016-16_rect.nc
#             filename = f'MCD19A1.{MODIS_tile}.061_01d_16day_HighVZA-MVC_QCfilt_{QC_option}_{yyyy}-{composite_nn:02d}_rect.nc'

#             file_fullpath = f'{dat_dir}/{filename}'
#             if not os.path.exists(file_fullpath):
#                 print(f'    File missing: {filename}')
#                 tile_files_missing += 1
#     if tile_files_missing == 0:
#         print(f'tile {MODIS_tile} - no files missing (all good)')   
#         tiles_complete.append(MODIS_tile)     
#     else:
#         print(f'tile {MODIS_tile} - {tile_files_missing} files missing')
#         tiles_to_reprocess.append(MODIS_tile)
    
#     print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#     print("")

# print(f'Step 5 DONE: Tiles with all 0.01° regridded files done: {tiles_complete}')
# print(f'             Tiles with missing 0.01° regridded files : {tiles_to_reprocess}')


# #%%
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # 6 - historic average gap-filled files
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# dat_basedir = '/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/HighVZA-MVC/1km_curv/lm-fill/'

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
        
#         #sample: MCD19A1.h12v04.061_1km_HighVZA-MVC_lmFill_QCfilt_CloudFree_hist_avg_12.nc
#         filename = f'MCD19A1.{MODIS_tile}.061_1km_HighVZA-MVC_lmFill_QCfilt_{QC_option}_hist_avg_{composite_nn:02d}.nc'
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

# #%%
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # 7 - gap-filled standard anomaly files
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# dat_basedir = '/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/HighVZA-MVC/1km_curv/std_anom/'

# tiles_complete = []
# tiles_to_reprocess = []
# print(f'QC filtering Option: {QC_option}')
# for MODIS_tile in MODIS_tiles:

#     tile_files_missing = 0
#     print("")
#     print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#     print(f'tile {MODIS_tile}')
#     dat_dir = os.path.join(dat_basedir, f'QCfilt_{QC_option}', MODIS_tile)
#     for yyyy in file_years:
#         for composite_nn in range(0, 23):

#             # skip files before March 2000
#             if ((yyyy == 2000) and np.isin(composite_nn, [0, 1, 2])):
#                 continue

#             # Skip processing if 'v01' is in MODIS_tile and composite_period is in [0, 1, 2, 19, 20, 21, 22]
#             if 'v01' in MODIS_tile and composite_nn in [0, 1, 2, 19, 20, 21, 22]:
#                 continue

#             # Skip processing if 'v02' is in MODIS_tile and composite_period is in [0, 1, 20, 21, 22]
#             if 'v02' in MODIS_tile and composite_nn in [0, 1, 20, 21, 22]:
#                 continue
            
#             #sample: MCD19A1.h12v04.061_1km_16day_HighVZA-MVC_stdAnom_QCfilt_CloudFree_2025-22.nc
#             filename = f'MCD19A1.{MODIS_tile}.061_1km_16day_HighVZA-MVC_stdAnom_QCfilt_{QC_option}_{yyyy}-{composite_nn:02d}.nc'

#             file_fullpath = f'{dat_dir}/{filename}'
#             if not os.path.exists(file_fullpath):
#                 print(f'    File missing: {filename}')
#                 tile_files_missing += 1
#     if tile_files_missing == 0:
#         print(f'tile {MODIS_tile} - no files missing (all good)')
#         tiles_complete.append(MODIS_tile)   
#     else:
#         print(f'tile {MODIS_tile} - {tile_files_missing} files missing')
#         tiles_to_reprocess.append(MODIS_tile)
    
#     print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#     print("")

# print(f'Step 7 DONE: Tiles with all gap-filled standard anomaly files done: {tiles_complete}')
# print(f'             Tiles with missing gap-filled standard anomaly files : {tiles_to_reprocess}')

# #%%
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # 8 - regridded standard anomaly files
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# dat_basedir = '/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/HighVZA-MVC/01d_rect/std_anom/'

# tiles_complete = []
# tiles_to_reprocess = []
# print(f'QC filtering Option: {QC_option}')
# for MODIS_tile in MODIS_tiles:

#     tile_files_missing = 0
#     print("")
#     print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#     print(f'tile {MODIS_tile}')
#     dat_dir = os.path.join(dat_basedir, f'QCfilt_{QC_option}', MODIS_tile)
#     for yyyy in file_years:
#         for composite_nn in range(0, 23):

#             # skip files before March 2000
#             if ((yyyy == 2000) and np.isin(composite_nn, [0, 1, 2])):
#                 continue

#             # Skip processing if 'v01' is in MODIS_tile and composite_period is in [0, 1, 2, 19, 20, 21, 22]
#             if 'v01' in MODIS_tile and composite_nn in [0, 1, 2, 19, 20, 21, 22]:
#                 continue

#             # Skip processing if 'v02' is in MODIS_tile and composite_period is in [0, 1, 20, 21, 22]
#             if 'v02' in MODIS_tile and composite_nn in [0, 1, 20, 21, 22]:
#                 continue
            
#             #sample: MCD19A1.h12v04.061_01d_16day_HighVZA-MVC_stdAnom_QCfilt_CloudFree_2025-21_rect.nc
#             filename = f'MCD19A1.{MODIS_tile}.061_01d_16day_HighVZA-MVC_stdAnom_QCfilt_{QC_option}_{yyyy}-{composite_nn:02d}_rect.nc'

#             file_fullpath = f'{dat_dir}/{filename}'
#             if not os.path.exists(file_fullpath):
#                 print(f'    File missing: {filename}')
#                 tile_files_missing += 1
#     if tile_files_missing == 0:
#         print(f'tile {MODIS_tile} - no files missing (all good)')  
#         tiles_complete.append(MODIS_tile)      
#     else:
#         print(f'tile {MODIS_tile} - {tile_files_missing} files missing')
#         tiles_to_reprocess.append(MODIS_tile)
    
#     print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#     print("")

# print(f'Step 8 DONE: Tiles with all 0.01° regridded standard anomaly files done: {tiles_complete}')
# print(f'             Tiles with missing 0.01° regridded standard anomaly files : {tiles_to_reprocess}')

# %%
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 9 - regridded historical average files
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

dat_basedir = '/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/HighVZA-MVC/01d_rect/'

tiles_complete = []
tiles_to_reprocess = []
print(f'QC filtering Option: {QC_option}')
for MODIS_tile in MODIS_tiles:

    tile_files_missing = 0
    print("")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(f'tile {MODIS_tile}')
    dat_dir = os.path.join(dat_basedir, f'QCfilt_{QC_option}', MODIS_tile)
    
    for composite_nn in range(0, 23):

        # Skip processing if 'v01' is in MODIS_tile and composite_period is in [0, 1, 2, 19, 20, 21, 22]
        if 'v01' in MODIS_tile and composite_nn in [0, 1, 2, 19, 20, 21, 22]:
            continue

        # Skip processing if 'v02' is in MODIS_tile and composite_period is in [0, 1, 20, 21, 22]
        if 'v02' in MODIS_tile and composite_nn in [0, 1, 20, 21, 22]:
            continue
        
        #sample: MCD19A1.h09v05.061_01d_HighVZA-MVC_QCfilt_CloudFree_hist_avg_21.nc
    
        filename = f'MCD19A1.{MODIS_tile}.061_01d_HighVZA-MVC_QCfilt_{QC_option}_hist_avg_{composite_nn:02d}.nc'

        file_fullpath = f'{dat_dir}/hist_avg/{filename}'
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

print(f'Step 8 DONE: Tiles with all 0.01° regridded historical average files done: {tiles_complete}')
print(f'             Tiles with missing 0.01° regridded historical average files : {tiles_to_reprocess}')

# # %%

# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # Other - regridded JJA annual average files
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# dat_basedir = '/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/HighVZA-MVC/01d_rect/'

# tiles_complete = []
# tiles_to_reprocess = []
# print(f'QC filtering Option: {QC_option}')
# for MODIS_tile in MODIS_tiles:

#     print("")
#     print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#     print(f'tile {MODIS_tile}')
#     dat_dir = os.path.join(dat_basedir, f'QCfilt_{QC_option}', MODIS_tile, 'annualized')
    
#     #sample: MCD19A1.h10v05.061_01d_annual_JJA_HighVZA-MVC_QCfilt_CloudFree.nc
#     file_fullpath = os.path.join(dat_dir, f'MCD19A1.{MODIS_tile}.061_01d_annual_JJA_HighVZA-MVC_QCfilt_{QC_option}.nc')


#     if os.path.exists(file_fullpath):
#         print(f'tile {MODIS_tile} - annualized JJA file exists (all good)')  
#         tiles_complete.append(MODIS_tile)      
#     else:
#         print(f'tile {MODIS_tile} - annualized JJA file missing')
#         tiles_to_reprocess.append(MODIS_tile)
    
#     print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#     print("")

# print(f'Step X DONE: Tiles with all 0.01° regridded annualized JJA files done: {tiles_complete}')
# print(f'             Tiles with missing 0.01° regridded annualized JJA files : {tiles_to_reprocess}')

# # %%

# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # Other - 1km JJA annual average files
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# dat_basedir = '/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/HighVZA-MVC/1km_curv/lm-fill/'

# tiles_complete = []
# tiles_to_reprocess = []
# print(f'QC filtering Option: {QC_option}')
# for MODIS_tile in MODIS_tiles:

#     print("")
#     print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#     print(f'tile {MODIS_tile}')
#     dat_dir = os.path.join(dat_basedir, f'QCfilt_{QC_option}', MODIS_tile, 'annualized')
    
#     #sample: MCD19A1.h10v05.061_1km_annual_JJA_HighVZA-MVC_QCfilt_CloudFree.nc
#     file_fullpath = os.path.join(dat_dir, f'MCD19A1.{MODIS_tile}.061_1km_annual_JJA_HighVZA-MVC_QCfilt_{QC_option}.nc')


#     if os.path.exists(file_fullpath):
#         print(f'tile {MODIS_tile} - annualized JJA file exists (all good)')  
#         tiles_complete.append(MODIS_tile)      
#     else:
#         print(f'tile {MODIS_tile} - annualized JJA file missing')
#         tiles_to_reprocess.append(MODIS_tile)
    
#     print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#     print("")

# print(f'Step X DONE: Tiles with all 0.01° regridded annualized JJA files done: {tiles_complete}')
# print(f'             Tiles with missing 0.01° regridded annualized JJA files : {tiles_to_reprocess}')
