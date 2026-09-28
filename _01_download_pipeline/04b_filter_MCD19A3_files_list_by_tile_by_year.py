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
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np

#########################################
# Define Global Filepaths
#########################################
#%%
# Output from 01a script 
master_files_list_dir = '/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A3/pre_proc/'

master_files_list_file = os.path.join(master_files_list_dir, 'all_MCD19A3_files_2000-2025.txt')

tile_files_list_basedir = '/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A3/pre_proc/ind_years/'

#########################################
# Define Global Variables
#########################################

#########################################
# Begin main
#########################################

tile_codes = [	
    # "h07v06", 
    # "h08v03",
    # "h08v04",
    # "h08v05",
    # "h08v06",
    # "h08v07",
    # "h09v02",
    # "h09v03",
    "h09v04", # US-NR1
    # "h09v05",
    # "h09v06",
    # "h09v07",
    # "h10v02",
    # "h09v08",
    # "h10v03",
    "h10v04", # US-Ne3
    # "h10v05",
    "h10v06", # OSBS
    # "h10v07",
    # "h10v08",
    "h11v02", # DEJU
    "h11v03", # Ca-Obs
    # "h11v04",
    # "h11v05",
    # "h11v06",
    # "h11v07",
    # "h11v08",
    # "h12v01",
    # "h12v02",
    # "h12v03",
    "h12v04", # US-UMB
    # "h12v05",
    # "h13v01",
    # "h13v02",
    # "h13v03",
    # "h13v04",
    # "h14v01",
    # "h14v02",
    # "h14v03",
    # "h14v04",
    # "h15v01",
    # "h15v02",
    # "h16v01",
    ]

years = np.arange(2000, 2026)

#%%
for tile_code in tile_codes:

    tile_files_list_dir = os.path.join(tile_files_list_basedir, tile_code)
    Path(tile_files_list_dir).mkdir(parents=True, exist_ok=True)
    print(f'Processing tile {tile_code}...')
    for yyyy in years:
        print(f'  Processing year {yyyy}...')
        # This is the output file with the filtered files list for the tile of interest
        tile_files_list_file = os.path.join(tile_files_list_dir, f'MCD19A3_files_{tile_code}_{yyyy}.txt')

        if os.path.exists(tile_files_list_file):
            print(f'File {tile_files_list_file} already exists, skipping tile {tile_code}')

        # Read in the master files list
        with open(master_files_list_file, 'r') as f:
            all_files = f.readlines()

        all_files = [line.rstrip('\n') for line in all_files]

        # Filter files by tile code
        filtered_files = [f.strip() for f in all_files if (tile_code in f and f'A{yyyy}' in f)]
        print(f'Found {len(filtered_files)} files for tile {tile_code} in {yyyy}')

        # Identify any dates with duplicate files for a given tile and 
        # retain only the file with the latest modification date

        # get array of dates represented by the files
        dates = []
        for file in filtered_files:
            basename = os.path.basename(file)
            parts = basename.split('.')
            if len(parts) > 1 and parts[1].startswith('A'):
                yyyyddd = parts[1][1:]  # Remove the 'A'
                year = int(yyyyddd[:4])
                doy = int(yyyyddd[4:])
                date = datetime(year, 1, 1) + timedelta(days=doy - 1)
                dates.append(date)

        # get array of file modification datetimes
        mod_dates = []
        for file in filtered_files:
            basename = os.path.basename(file)
            parts = basename.split('.')
            if len(parts) > 2:
                mod_str = parts[-2]
                try:
                    mod_date = datetime.strptime(mod_str, "%Y%j%H%M%S")
                    mod_dates.append(mod_date)
                except ValueError:
                    pass

        # Identify the first date and last date in the list
        start_date = min(dates)
        end_date = max(dates)

        # Initialize to_keep flags for all files
        to_keep = [True] * len(filtered_files)

        # Create a mapping from date to indices in the filtered_files list

        date_to_indices = defaultdict(list)
        for idx, date in enumerate(dates):
            date_to_indices[date].append(idx)

        # Loop through all dates from start_date to end_date
        current_date = start_date
        while current_date <= end_date:
            indices = date_to_indices.get(current_date, [])
            if len(indices) > 1:
                print(f'Duplicate files found for date {current_date.strftime("%Y-%m-%d")}:')
                # More than one file for this date, find the one with the latest mod_date
                mod_dates_for_date = [mod_dates[i] for i in indices]
                max_mod_idx = mod_dates_for_date.index(max(mod_dates_for_date))
                # Set to_keep False for all except the latest
                for i, idx in enumerate(indices):
                    if i != max_mod_idx:
                        print(f'   {os.path.basename(filtered_files[idx])} is duplicate, setting flag to discard')
                        to_keep[idx] = False
                    else:
                        print(f'   {os.path.basename(filtered_files[idx])} is kept as latest')
            current_date += timedelta(days=1)

        # Filter the files to only those to keep
        filtered_files = [f for f, keep in zip(filtered_files, to_keep) if keep]
        print(f'{len(filtered_files)} files remain in list for tile {tile_code} in {yyyy}')

        # Write to file
        with open(tile_files_list_file, 'w') as f:
            for file in filtered_files:
                f.write(file + '\n')


# %%
