#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Script to generate sanity check plots for MODIS MCD19A1 CV-MVC composite files
# Author: Lewis Kunik - University of Utah
# Contact: lewis.kunik@utah.edu
#

import os
import sys
import glob
from pathlib import Path
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
from cartopy.io import img_tiles
import cartopy.io.shapereader as shpreader
import cartopy.feature as cfeature

# Define output base directory
plot_basedir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/plots/maps/sanity_check/'

# Set up map parameters
tiler = img_tiles.GoogleTiles(style='satellite')
crs = tiler.crs
alpha = 0.8
transform = ccrs.PlateCarree()

# Load state boundaries
fn = shpreader.natural_earth(
   resolution='10m', category='cultural', 
   name='admin_1_states_provinces',
)
reader = shpreader.Reader(fn)
states = [x for x in reader.records() if x.attributes["admin"] == "United States of America"]
states_geom = cfeature.ShapelyFeature([x.geometry for x in states], ccrs.PlateCarree())

tiler_zoom = 5

def get_tile_plot_extent(dat_xr_curv):
    lat_min = float(dat_xr_curv['lat'].min())
    lat_max = float(dat_xr_curv['lat'].max())
    lon_min = float(dat_xr_curv['lon'].min())
    lon_max = float(dat_xr_curv['lon'].max())
    plot_extent = [lon_min, lon_max, lat_min, lat_max]

    # Special case if some lon values cross the 180 meridian
    if ((lon_min < -175) & (lon_max > 100)):
        lon_neg = dat_xr_curv.lon.values.copy()
        lon_neg[lon_neg > 0] = np.nan
        plot_extent = [-179.99, np.nanmax(lon_neg), lat_min, lat_max]

    return plot_extent

def test_plot(dat_da, plot_title, zlab, plot_extent=None, plot_filename=None, plot_min=None, plot_max=None):
    if plot_extent is None:
        plot_extent = get_tile_plot_extent(dat_da)
    if plot_min is None:
        plot_min = float(dat_da.min())
    if plot_max is None:
        plot_max = float(dat_da.max())
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': crs})
    ax.set_extent(plot_extent, crs=ccrs.PlateCarree())
    ax.add_image(tiler, tiler_zoom, interpolation='none')
    ax.add_feature(states_geom, facecolor="none", edgecolor="black", linewidth=1)
    
    dat_da.plot(ax=ax, x='lon', y='lat', transform=transform, 
                alpha=alpha, cmap='inferno_r',
                vmin=plot_min, vmax=plot_max,
                cbar_kwargs={'orientation': 'vertical',
                            'pad': 0.05,
                            'label': zlab,
                            'shrink': 0.75})

    plt.title(plot_title, fontsize=18)
    plt.xlabel('lon', fontsize=15)
    plt.ylabel('lat', fontsize=14)

    if plot_filename is not None:
        if not os.path.exists(os.path.dirname(plot_filename)):
            Path(os.path.dirname(plot_filename)).mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_filename, format="png", bbox_inches="tight", dpi=150)
        plt.close('all')
    else:
        plt.show()

def extract_tile_from_filename(filename):
    """Extract MODIS tile code (e.g., 'h09v02') from filename"""
    basename = os.path.basename(filename)
    # Look for pattern hXXvXX
    parts = basename.split('.')
    for part in parts:
        if part.startswith('h') and 'v' in part and len(part) == 6:
            return part
    return None

def plot_composite_file(nc_file):
    """Generate sanity check plot for a single composite file"""
    
    # Extract tile code from filename
    tile_code = extract_tile_from_filename(nc_file)
    if tile_code is None:
        print(f"Could not extract tile code from {os.path.basename(nc_file)}, skipping")
        return
    
    # Set up output directory and filename
    plot_dir = os.path.join(plot_basedir, tile_code)
    Path(plot_dir).mkdir(parents=True, exist_ok=True)
    
    basename = os.path.splitext(os.path.basename(nc_file))[0]
    plot_filename_NDVI = os.path.join(plot_dir, f'{basename}_NDVI.png')
    plot_filename_CCI = os.path.join(plot_dir, f'{basename}_CCI.png')
    
    # # Check if plot already exists
    # if os.path.exists(plot_filename_NDVI):
    #     print(f"Plot already exists: {plot_filename_NDVI}, skipping")
    #     return
    
    # Load the composite file
    try:
        ds = xr.open_dataset(nc_file)
        
        # Check if required reflectance bands exist
        if 'Sur_refl1' not in ds or 'Sur_refl11' not in ds or 'Sur_refl2' not in ds:
            print(f"  Missing Sur_refl1, Sur_refl11, or Sur_refl2 in {basename}, cannot calculate NDVI or CCI")
            ds.close()
            return
        
        # Calculate NDVI
        NDVI = (ds['Sur_refl2'] - ds['Sur_refl1']) / (ds['Sur_refl2'] + ds['Sur_refl1'])
        CCI = (ds['Sur_refl11'] - ds['Sur_refl1']) / (ds['Sur_refl11'] + ds['Sur_refl1'])
        
        # Squeeze time dimension if it exists and has length 1
        if 'time' in ds.dims and len(ds.time) == 1:
            NDVI = NDVI.squeeze('time')
            CCI = CCI.squeeze('time')
        
        # Get plot title from file attributes or filename
        if 'date_string' in ds.attrs:
            plot_title = f"{tile_code} - {ds.attrs['date_string']}"
        else:
            plot_title = f"{tile_code} - {basename[-10:-4]}"
        
        # Make the plot
        print(f"  Creating plot: {os.path.basename(plot_filename_NDVI)}")
        test_plot(NDVI, plot_title, 'NDVI', 
                 plot_filename=plot_filename_NDVI,
                 plot_min=0, plot_max=1)
        
        ds.close()
        print(f"  Successfully created NDVI map: {plot_filename_NDVI}")

                # Make the plot
        print(f"  Creating plot: {os.path.basename(plot_filename_CCI)}")
        test_plot(CCI, plot_title, 'CCI', 
                 plot_filename=plot_filename_CCI,
                 plot_min=-0.3, plot_max=0.2)
        
        ds.close()
        print(f"  Successfully created CCI map: {plot_filename_CCI}")
        
    except Exception as e:
        print(f"  ERROR processing {os.path.basename(nc_file)}: {e}")

if __name__ == "__main__":
    
    # Get input file(s) from command line or glob pattern
    if len(sys.argv) > 1:
        # User provided file path(s) or glob pattern
        input_pattern = sys.argv[1]
        nc_files = sorted(glob.glob(input_pattern))
    else:
        print("Usage: python script.py <path_to_nc_file_or_glob_pattern>")
        print("Example: python script.py '/path/to/composites/MCD19A1*.nc'")
        sys.exit(1)
    
    if len(nc_files) == 0:
        print(f"No files found matching pattern: {input_pattern}")
        sys.exit(1)
    
    print(f"Found {len(nc_files)} files to process")
    
    # Process each file
    for i, nc_file in enumerate(nc_files):
        print(f"\n[{i+1}/{len(nc_files)}] Processing: {os.path.basename(nc_file)}")
        plot_composite_file(nc_file)
    
    print("\nDone!")