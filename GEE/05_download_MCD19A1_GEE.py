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
import os
from tracemalloc import start

# numerical/data management packages
import numpy as np
import pandas as pd

# plotting packages
import matplotlib.pyplot as plt

# date and time packages
from datetime import datetime
import time


# Google Earth Engine packages
import ee
import geemap



#########################################
# Define Global Filepaths
#########################################
#%%
out_dir = '/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/from_GEE'

########################################
# Define Global Variables and constants
#########################################

### UNCOMMENT LINES FOR EACH CITY AREA TO USE A DIFFERENT BOUNDING BOX ###
# Bounding box (min_lon, min_lat, max_lon, max_lat)


bbox_coords = (-140.0151, 40, -104.4217, 50)  # test overlap area encompassing multiple tiles

# lon_h09v04_min = -140.0151
# lon_h09v04_max = -104.4217
# lat_h09v04_min = 40.0000
# lat_h09v04_max = 50.0000


# lon_h11v04_min = -108.9007   
# lon_h11v04_max = -78.3136
# lat_h11v04_min = 40.0000
# lat_h11v04_max = 50.0000


###########################
# Define Global Functions
#########################################


def authenticate_gee():
    """Authenticate and initialize Google Earth Engine"""
    try:
        ee.Initialize()
        print("GEE already authenticated")
    except:
        print("Authenticating GEE...")
        ee.Authenticate()
        ee.Initialize()
        print("GEE authentication complete")

def download_mcd19a1(bbox_coords, start_date, end_date, output_path):
    """
    Download MODIS MCD19A1 data for specified region and time period

    Args:
        bbox_coords: tuple of (min_lon, min_lat, max_lon, max_lat)
        start_date: string in 'YYYY-MM-DD' format
        end_date: string in 'YYYY-MM-DD' format  
        output_path: string path for output netcdf file
    """
    
    # Define the bounding box geometry
    min_lon, min_lat, max_lon, max_lat = bbox_coords
    bbox = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])

    # Load MODIS MCD19A1 dataset
    modis_maiac = ee.ImageCollection('MODIS/061/MCD19A1_GRANULES')
    
    # Filter by date and region
    filtered = modis_maiac.filterDate(start_date, end_date).filterBounds(bbox)
    
    # Check if data exist
    collection_size = filtered.size().getInfo()
    if collection_size == 0:
        print(f"No data found for the specified time period: {start_date} to {end_date}")
        return None
    
    print(f"Found {collection_size} images in the collection")

    download_bands = ['Sur_refl1', 
                      'Sur_refl2', 
                      'Sur_refl3', 
                      'Sur_refl11',
                      'Sigma_BRFn1',
                      'Sigma_BRFn2',
                      'Status_QA',
                      'cosSZA',
                      'cosVZA',
                      'RelAZ',
                      'Scattering_Angle',
                      'SAZ',
                      'VAZ',
                      'Glint_Angle']
    

    # Select the specified bands
    filtered_bands = filtered.select(download_bands)

    # Create a median composite (or use mean, depending on your needs)
    # For daily data, you might want to use median to reduce cloud/bad pixel effects
    composite = filtered_bands.median()
    
    # Clip to bounding box
    maiac_clipped = composite.clip(bbox)
    
    # Reproject to EPSG:4326 (WGS84) with 500m pixel size
    # Note: 500m ≈ 0.004491556 degrees at the equator
    maiac_reprojected = maiac_clipped.reproject(
        crs='EPSG:4326',
        scale=1000  # Keep original 1000m resolution
    )
    
    print("Reprojected to EPSG:4326")
    
    # Export to Google Drive first, then download
    task = ee.batch.Export.image.toDrive(
        image=maiac_reprojected,  # Use reprojected image
        description='MODIS_MCD19A1_MAIAC_Export',
        folder='GEE_Exports',
        fileNamePrefix=f'modis_mcd19a1_{start_date.replace("-", "")}_{end_date.replace("-", "")}',
        scale=1000,  # MCD19A1 native resolution
        region=bbox,
        fileFormat='GeoTIFF',
        crs='EPSG:4326'  # Explicitly specify CRS for export
    )
    
    print("Starting export task...")
    task.start()
    
    # Monitor task status
    import time
    while task.active():
        print(f"Task status: {task.status()['state']}")
        time.sleep(10)
    
    print(f"Export completed with status: {task.status()['state']}")
    
    # For direct download without Drive, use geemap
    print("Downloading data using geemap...")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Download the image with explicit CRS specification
    geemap.ee_export_image(
        maiac_reprojected,  # Use reprojected image
        filename=output_path,
        scale=1000,
        region=bbox,
        file_per_band=False,
        crs='EPSG:4326'  # Explicitly specify CRS for geemap export
    )

    return maiac_reprojected, bbox


#%%

#########################################
# Begin Main
#########################################

# keep track of elapsed time 
start_time = time.time()


# Sample time period (2020) - adjust as needed
start_date = '2020-01-01'
end_date = '2020-12-31'

# Output path
output_path = os.path.join(out_dir, f'MCD19A1_2020_test_BBlarge_h09v04.tif')  # Note: geemap exports as GeoTIFF

print(f"Starting MODIS MAIAC data download for test bounding box...")
print(f"Bounding box (in degrees lat/lon): {bbox_coords}")
print(f"Date range: {start_date} to {end_date}")

# Authenticate Google Earth Engine
ee.Authenticate() # UNCOMMENT AT FIRST RUN IN A SESSION TO AUTHENTICATE WITH GEE
ee.Initialize(project='ee-lewiskunik') # must be linked to a project with a GEE account - I used mine

# Download the data
lc_image, bbox = download_mcd19a1(bbox_coords, start_date, end_date, output_path)

if lc_image is not None:
   print(f"Data downloaded successfully to: {output_path}")

else:
   print("Failed to download data")


# Print elapsed time
elapsed_time = time.time() - start_time
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"Elapsed time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")


# %%
