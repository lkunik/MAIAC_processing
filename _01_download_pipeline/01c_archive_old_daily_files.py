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
import sys
from pathlib import Path
import glob

#########################################
# Define Global Filepaths
#########################################
#%%

# location of hdf files
hdf_dir = '/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/pre_proc/hdf'
obsolete_hdf_archive_dir = os.path.join(hdf_dir, 'archive')

# location of raw nc files 
raw_nc_dir = '/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/pre_proc/raw_nc/'
obsolete_nc_archive_dir = os.path.join(raw_nc_dir, 'archive')

# Output from 01a script
master_files_list_dir = '/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/pre_proc/'
master_files_list_file = os.path.join(master_files_list_dir, 'all_MCD19A1_files_2000-2025.txt')

#########################################
# Define Global Variables
#########################################

tile_code = 'h10v04' # this is the tile of interest
# This is the output file with the filtered files list for the tile of interest
tile_files_list_file = os.path.join(master_files_list_dir, f'MCD19A1_files_{tile_code}_2000-2025.txt')

#########################################
# Begin main
#########################################
#%%

# List existing hdf files
existing_hdf_full_filepaths = sorted(glob.glob(os.path.join(hdf_dir, '*.hdf')))
print(f'Number of existing HDF files: {len(existing_hdf_full_filepaths)}')

existing_hdf_files = [os.path.basename(f) for f in existing_hdf_full_filepaths]

# # Read in the master files list (all tiles)
# with open(master_files_list_file, 'r') as f:
#    all_files = f.readlines()

# all_files = [line.rstrip('\n') for line in all_files]
# print(f'Number of files found in {os.path.basename(master_files_list_file)}: {len(all_files)}')

# Read in the master tile-specific files list
with open(tile_files_list_file, 'r') as f:
   all_earthaccess_repo_files = f.readlines()

all_earthaccess_repo_files = [line.rstrip('\n') for line in all_earthaccess_repo_files]
print(f'Number of files found in {os.path.basename(tile_files_list_file)}: {len(all_earthaccess_repo_files)}')

# get basenames only
all_earthaccess_repo_file_basenames = [os.path.basename(f) for f in all_earthaccess_repo_files]

# get the existing files which are not present in the earthaccess files list 
# (i.e. the files which have updated versions on NASA's repo)
obsolete_hdf_files = [f for f in existing_hdf_files if f not in all_earthaccess_repo_file_basenames]
print(f'Number of obsolete HDF files to archive: {len(obsolete_hdf_files)}')


#%%

# PERFORM ARCHIVING - MOVE OLD HDF FILES TO ARCHIVE DIRECTORY

# Move obsolete files to the archive directory
os.makedirs(obsolete_hdf_archive_dir, exist_ok=True)
for filename in obsolete_hdf_files:
    src = os.path.join(hdf_dir, filename)
    dst = os.path.join(obsolete_hdf_archive_dir, filename)
    if os.path.isfile(src):
        os.rename(src, dst)

#%%

# %%

# List existing netcdf files
existing_netcdf_full_filepaths = sorted(glob.glob(os.path.join(raw_nc_dir, '*.nc')))
print(f'Number of existing NetCDF files: {len(existing_netcdf_full_filepaths)}')

# convert to basenames and remove extensions
existing_netcdf_files = [os.path.basename(f) for f in existing_netcdf_full_filepaths]
existing_netcdf_files_no_ext = [os.path.splitext(f)[0] for f in existing_netcdf_files]

# Filter files by tile code
existing_netcdf_files_no_ext_tile_filt = [f.strip() for f in existing_netcdf_files_no_ext if tile_code in f]
print(f'Number of nc files found in {raw_nc_dir} for tile {tile_code}: {len(existing_netcdf_files_no_ext_tile_filt)}')

all_earthaccess_repo_files_no_ext = [os.path.splitext(f)[0] for f in all_earthaccess_repo_file_basenames]
print(f'Number of files found in {os.path.basename(tile_files_list_file)}: {len(all_earthaccess_repo_files_no_ext)}')

# get the existing files which are not present in the earthaccess files list 
# (i.e. the files which have updated versions on NASA's repo)
obsolete_nc_files = [f for f in existing_netcdf_files_no_ext_tile_filt if f not in all_earthaccess_repo_files_no_ext]
print(f'Number of obsolete NetCDF files to archive: {len(obsolete_nc_files)}')

#%%

# PERFORM ARCHIVING - MOVE OLD NETCDF FILES TO ARCHIVE DIRECTORY

# Move obsolete files to the archive directory
os.makedirs(obsolete_nc_archive_dir, exist_ok=True)
for filename in obsolete_nc_files:
    src = os.path.join(raw_nc_dir, f'{filename}.nc')
    dst = os.path.join(obsolete_nc_archive_dir, f'{filename}.nc')
    if os.path.isfile(src):
        os.rename(src, dst)

# %%
