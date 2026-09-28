#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Lewis Kunik - University of Utah
# Contact: lewis.kunik@utah.edu
#
# Created on Tue Oct 24 2023
#
#

#%%
import ee
import os
from datetime import datetime, timedelta
import requests
import time
from pathlib import Path

#%%
# Initialize Earth Engine
# ee.Authenticate()
ee.Initialize()

#%%
# Base directory for saving files
base_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/MODIS/Reflectance_MOD09/from_GEE'
# Site coordinates, date ranges, and save paths
# Site coordinates, date ranges, and save paths
site_info = {
    # 'US-NR1': {
    #     'coords': (-105.5464, 40.0329),
    #     'start': '2017-01-01',
    #     'end': '2018-12-31'
    # },
    # 'OSBS': {
    #     'coords': (-81.993431, 29.689282),
    #     'start': '2021-01-01',
    #     'end': '2022-12-31'
    # },
    # 'Ca-Obs': {
    #     'coords': (-105.1178, 53.9872),
    #     'start': '2018-01-01',
    #     'end': '2022-12-31'
    # },
    # 'DEJU': {
    #     'coords': (-145.75136, 63.88112),
    #     'start': '2019-01-01',
    #     'end': '2021-12-31'
    # },
    'US-UMB': {
        'coords': (-84.7138, 45.5598),
        'start': '2018-01-01',
        'end': '2019-12-31'
    },
    'US-Ne3': {
        'coords': (-96.4397, 41.1797),
        'start': '2018-01-01',
        'end': '2019-12-31'
    }
}



def download_modocga(site_name, lon, lat, start_date, end_date):
    """Download MODOCGA bands 8-12 for a single site"""
    
    start_time = time.time()
    site_dir = os.path.join(base_dir, site_name)
    Path(site_dir).mkdir(parents=True, exist_ok=True)
    
    # Define point geometry
    point = ee.Geometry.Point([lon, lat])
    
    # MYDOCGA bands 11, 12, QC bits
    bands = ['sur_refl_b11', 'sur_refl_b12', 'QC_b8_15_1km']
    
    # Get MYDOCGA collection (version 6.1)
    collection = ee.ImageCollection('MODIS/006/MYDOCGA') \
        .filterDate(start_date, end_date) \
        .filterBounds(point) \
        .select(bands)
    
    # Get list of images
    image_list = collection.toList(collection.size())
    n_images = image_list.size().getInfo()
    
    print(f'\n{site_name} MYDOCGA: Found {n_images} images')
    
    # Download each image
    for i in range(n_images):
        image = ee.Image(image_list.get(i))
        
        # Get date from image
        date_str = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
        
        # Define output filename
        out_file = os.path.join(site_dir, f'MYDOCGA_{site_name}_{date_str}.tif')
        
        # Skip if already exists
        if os.path.exists(out_file):
            if i % 100 == 0:
                checkpoint_time = time.time()
                elapsed = checkpoint_time - start_time
                start_time = checkpoint_time
                print(f'  {i}/{n_images}: {date_str} already exists - Elapsed {elapsed:.2f} seconds')
            continue
        
        # Export parameters
        region = point.buffer(1).bounds()
        
        try:
            # Get download URL
            url = image.getDownloadURL({
                'region': region,
                'scale': 1000,  # MYDOCGA native resolution
                'format': 'GEO_TIFF'
            })
            
            # Download file
            response = requests.get(url)
            
            if response.status_code == 200:
                with open(out_file, 'wb') as f:
                    f.write(response.content)
                
                if i % 50 == 0:
                    checkpoint_time = time.time()
                    elapsed = checkpoint_time - start_time
                    start_time = checkpoint_time
                    print(f'  {i}/{n_images}: Downloaded {date_str} - Elapsed {elapsed:.2f} seconds')
            else:
                print(f'  ERROR: {date_str} - Status {response.status_code}')
                
        except Exception as e:
            print(f'  ERROR: {date_str} - {str(e)}')
            continue

# Process each site
for site_name, info in site_info.items():
    lon, lat = info['coords']
    start_date = site_info[site_name]['start']
    end_date = site_info[site_name]['end']

    # Download MYDOCGA (bands 11, 12, QC bits)
    download_modocga(site_name, lon, lat, start_date, end_date)

print('\nAll downloads complete!')