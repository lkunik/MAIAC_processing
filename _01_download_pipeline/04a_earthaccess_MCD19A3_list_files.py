#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Lewis Kunik - University of Utah
# Contact: lewis.kunik@utah.edu
#


# See tutorial at https://github.com/nasa/LPDAAC-Data-Resources/blob/main/python/tutorials/earthaccess_introduction.ipynb

#########################################
# Load packages
#########################################
#%%
# File system packages
import os  # operating system library
import sys
from pathlib import Path


# numerical/data management packages
import numpy as np
import xarray as xr  # for multi dimensional data

# time and date packages
import time
from datetime import datetime as dt  # date time library
from datetime import timedelta

# Geospatial packages
import geopandas as gpd

# Earth Access packages
import earthaccess

#########################################
# Define Global Filepaths
#########################################
#%%
# download_dir = '/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A3/pre_proc/hdf'

# directory to save text file with list of all available files on Earthdata for this product
files_list_dir = '/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A3/pre_proc/'

# output text file with list of files
files_list_file = os.path.join(files_list_dir, 'all_MCD19A3_files_2000-2025.txt')

#########################################
# Define Global Variables
#########################################

files_years = np.arange(2000, 2026)  # Years of interest

#########################################
# Begin main
#########################################
#%%


# Need to authenticate with Earthdata Login
auth = earthaccess.login(persist = True)
# are we authenticated?
print(f'auth.authenticated: {auth.authenticated}' )

# %%

# Search for collections with keyword 'MCD19A3'
collections_req = earthaccess.search_datasets(
   provider='LPCLOUD',    # LPCLOUD is the LP DAAC Archive in Earthdata Cloud
   keyword='MCD19A3',
   version='061'
)

collection = collections_req[0]    # Get the first earthaccess DataCollection in the list
collection

collections_info = [{n:[c.summary()['short-name'], c.summary()['concept-id'], c.summary()['version']]} for n, c in enumerate(collections_req)]
collections_info

#%%
# Check if files_list_file is empty
if os.path.exists(files_list_file):
   if os.path.getsize(files_list_file) == 0:
      print(f"{files_list_file} is empty.")
   else:
      print(f"{files_list_file} is not empty.")
      with open(files_list_file, 'w') as f:
         pass
else:
   print(f"{files_list_file} does not exist.")


# %%
# Claude wrote this
for file_year in files_years:
   start_date = dt(file_year, 1, 1)
   end_date = dt(file_year + 1, 1, 1) - timedelta(days=1)

   print(f'Year: {file_year} - Start Date: {start_date} - End Date: {end_date}')
   # Build query
   granules_request = earthaccess.search_data(
      short_name='MCD19A3D', # This is where you specify the product - MODIFY here
      version='061', # product version, most MODIS products are at version 6.1, i.e. '061'
      provider='LPCLOUD',
      temporal=(start_date, end_date), # Define the start/end date of the search
      count=-1 # -1 means return all results (default is 10)
   )
   print(f'granules_request is a {type(granules_request)} of {type(granules_request[0])}')

   print(f'found {len(granules_request)} granules')
   #%%


   # #%%

   # # Test
   # granule = granules_request[0]
   # granule.data_links()


   #%%
   # Write the links to the files_list_file
   with open(files_list_file, 'a') as f:
      for granule in granules_request:
         links = granule.data_links()
         for link in links:
            print(f'writing {link}')
            f.write(link + '\n')
