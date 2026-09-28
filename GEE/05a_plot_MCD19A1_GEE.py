#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Script to load and plot MODIS MCD19A1 NDVI from downloaded TIF files
#

#%%
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from datetime import datetime
import os
import re


#########################################
# Define Global Filepaths
#########################################
#%%
dat_dir = '/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/from_GEE/'
plot_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/MAIAC/plots/'
csv_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/LAI/data/'
csv_file = os.path.join(csv_dir, '231116_all_site_metadata_for_publication_1_0.90_all_pass.csv')

# Create output directories if they don't exist
os.makedirs(plot_dir, exist_ok=True)

#%%

def load_mcd19a1_data(tif_path):
    """
    Load MODIS MCD19A1 data from a multiband GeoTIFF
    
    Args:
        tif_path: path to the GeoTIFF file
    
    Returns:
        dictionary with band data and metadata
    """
    
    print(f"Loading MODIS MCD19A1 data from: {os.path.basename(tif_path)}")
    
    with rasterio.open(tif_path) as src:
        # Read all bands
        data = src.read()
        
        # Get band descriptions
        try:
            band_descriptions = [src.descriptions[i] if src.descriptions[i] is not None 
                               else f"band_{i+1}" for i in range(src.count)]
        except:
            band_descriptions = [f"band_{i+1}" for i in range(src.count)]
        
        print(f"Found {src.count} bands")
        print(f"Spatial dimensions: {src.height} x {src.width} pixels")
        print(f"Pixel size: {src.res[0]:.1f} x {src.res[1]:.1f} degrees")
        print(f"Bands: {band_descriptions}")
        
        # Create band dictionary for easy access
        band_dict = {}
        for i, desc in enumerate(band_descriptions):
            band_dict[desc] = data[i, :, :]
        
        # Get spatial metadata
        transform = src.transform
        crs = src.crs
        bounds = src.bounds
        
        metadata = {
            'transform': transform,
            'crs': crs,
            'bounds': bounds,
            'shape': (src.height, src.width),
            'band_descriptions': band_descriptions
        }
    
    return band_dict, metadata

def compute_ndvi(red_band, nir_band, scale_factor=0.0001):
    """
    Compute NDVI from red and NIR surface reflectance bands
    
    Args:
        red_band: red surface reflectance band (Sur_refl1)
        nir_band: NIR surface reflectance band (Sur_refl2) 
        scale_factor: scaling factor for reflectance values (default 0.0001)
    
    Returns:
        NDVI array with invalid values masked
    """
    
    # Apply scale factor to convert to reflectance (0-1)
    red = red_band.astype(np.float64) * scale_factor
    nir = nir_band.astype(np.float64) * scale_factor
    
    # Create masks for valid data
    # MODIS valid range for surface reflectance is typically -0.1 to 1.6
    valid_red = (red >= -0.1) & (red <= 1.6)
    valid_nir = (nir >= -0.1) & (nir <= 1.6)
    valid_mask = valid_red & valid_nir
    
    # Compute NDVI
    denominator = nir + red
    
    # Avoid division by zero
    ndvi = np.full_like(red, np.nan)
    
    # Only compute NDVI where denominator is not zero and data is valid
    valid_calc = valid_mask & (np.abs(denominator) > 1e-6)
    ndvi[valid_calc] = (nir[valid_calc] - red[valid_calc]) / denominator[valid_calc]
    
    # Apply reasonable NDVI bounds (-1 to 1)
    ndvi[(ndvi < -1) | (ndvi > 1)] = np.nan
    
    return ndvi, valid_mask

def plot_ndvi_map(ndvi_array, metadata, title=None, save_path=None):
    """
    Create a map plot of NDVI
    
    Args:
        ndvi_array: 2D array of NDVI values
        metadata: dictionary with spatial metadata
        title: optional title for the plot
        save_path: optional path to save the plot
    """
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create extent for plotting
    bounds = metadata['bounds']
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    
    # Create custom colormap for NDVI
    colors_list = ['#8B4513', '#D2B48C', '#F5DEB3', '#FFFF00', '#ADFF2F', '#32CD32', '#006400']
    n_bins = 100
    cmap = colors.LinearSegmentedColormap.from_list('ndvi', colors_list, N=n_bins)
    
    # Plot NDVI
    im = ax.imshow(ndvi_array, extent=extent, cmap=cmap, vmin=-0.2, vmax=1.0, 
                   interpolation='nearest', origin='upper')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('NDVI', fontsize=12)
    
    # Customize plot
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=14, pad=20)
    else:
        ax.set_title('MODIS MCD19A1 NDVI', fontsize=14, pad=20)
    
    # Add grid
    ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"NDVI map saved to: {save_path}")
    else:
        plt.show()

def plot_ndvi_histogram(ndvi_array, title=None, save_path=None):
    """
    Create a histogram of NDVI values
    
    Args:
        ndvi_array: 2D array of NDVI values
        title: optional title for the plot
        save_path: optional path to save the plot
    """
    
    # Flatten array and remove NaN values
    ndvi_flat = ndvi_array.flatten()
    ndvi_valid = ndvi_flat[~np.isnan(ndvi_flat)]
    
    if len(ndvi_valid) == 0:
        print("No valid NDVI values found")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create histogram
    n, bins, patches = ax.hist(ndvi_valid, bins=50, density=True, alpha=0.7, 
                              color='darkgreen', edgecolor='black', linewidth=0.5)
    
    # Customize plot
    ax.set_xlabel('NDVI', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=14, pad=20)
    else:
        ax.set_title('MODIS MCD19A1 NDVI Distribution', fontsize=14, pad=20)
    
    # Add statistics
    stats_text = f'Mean: {np.mean(ndvi_valid):.3f}\n'
    stats_text += f'Median: {np.median(ndvi_valid):.3f}\n'
    stats_text += f'Std: {np.std(ndvi_valid):.3f}\n'
    stats_text += f'Valid pixels: {len(ndvi_valid):,}'
    
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"NDVI histogram saved to: {save_path}")
    else:
        plt.show()

def analyze_ndvi_statistics(ndvi_array, mask_array=None):
    """
    Calculate and print NDVI statistics
    
    Args:
        ndvi_array: 2D array of NDVI values
        mask_array: optional mask for valid pixels
    """
    
    if mask_array is not None:
        ndvi_valid = ndvi_array[mask_array & ~np.isnan(ndvi_array)]
    else:
        ndvi_valid = ndvi_array[~np.isnan(ndvi_array)]
    
    if len(ndvi_valid) == 0:
        print("No valid NDVI values found")
        return
    
    print("\nNDVI Statistics:")
    print(f"Valid pixels: {len(ndvi_valid):,}")
    print(f"Total pixels: {ndvi_array.size:,}")
    print(f"Valid percentage: {len(ndvi_valid)/ndvi_array.size*100:.1f}%")
    print(f"Mean NDVI: {np.mean(ndvi_valid):.3f}")
    print(f"Median NDVI: {np.median(ndvi_valid):.3f}")
    print(f"Standard deviation: {np.std(ndvi_valid):.3f}")
    print(f"Min NDVI: {np.min(ndvi_valid):.3f}")
    print(f"Max NDVI: {np.max(ndvi_valid):.3f}")
    print(f"25th percentile: {np.percentile(ndvi_valid, 25):.3f}")
    print(f"75th percentile: {np.percentile(ndvi_valid, 75):.3f}")

#%%

def process_mcd19a1_file(tif_path, site_name=None):
    """
    Process a single MCD19A1 file to compute and plot NDVI
    
    Args:
        tif_path: path to the MCD19A1 TIF file
        site_name: optional site name for labeling
    """
    
    # Load the data
    band_dict, metadata = load_mcd19a1_data(tif_path)
    
    # Check if required bands are present
    if 'Sur_refl1' not in band_dict or 'Sur_refl2' not in band_dict:
        print("Error: Required bands (Sur_refl1, Sur_refl2) not found in the data")
        print(f"Available bands: {list(band_dict.keys())}")
        return
    
    # Compute NDVI
    ndvi, valid_mask = compute_ndvi(band_dict['Sur_refl1'], band_dict['Sur_refl2'])
    
    # Generate base filename for outputs
    tif_basename = os.path.splitext(os.path.basename(tif_path))[0]
    
    # Create title
    if site_name:
        title_base = f'MODIS MCD19A1 NDVI - {site_name}'
    else:
        title_base = f'MODIS MCD19A1 NDVI - {tif_basename}'
    
    # Plot NDVI map
    map_save_path = os.path.join(plot_dir, f"{tif_basename}_ndvi_map.png")
    plot_ndvi_map(ndvi, metadata, title=title_base, save_path=map_save_path)
    
    # Plot NDVI histogram
    hist_save_path = os.path.join(plot_dir, f"{tif_basename}_ndvi_histogram.png")
    plot_ndvi_histogram(ndvi, title=f'{title_base} - Distribution', save_path=hist_save_path)
    
    # Print statistics
    analyze_ndvi_statistics(ndvi, valid_mask)
    
    return ndvi, valid_mask, metadata

#%%



# Option 1: Process a specific file
tif_file = 'MCD19A1_2020_test_BB.tif'  # Update this to your actual filename
tif_path = os.path.join(dat_dir, tif_file)

if os.path.exists(tif_path):
    print(f"Processing single file: {tif_file}")
    ndvi, valid_mask, metadata = process_mcd19a1_file(tif_path)
else:
    print(f"File not found: {tif_path}")
    print(f"Available files in {dat_dir}:")
    for f in os.listdir(dat_dir):
        if f.endswith('.tif'):
            print(f"  {f}")

# Option 2: Process multiple files (if you have multiple MCD19A1 files)
# Uncomment the code below if you want to process multiple files

# # Get all TIF files in the directory
# tif_files = [f for f in os.listdir(dat_dir) if f.endswith('.tif')]
# 
# for tif_file in tif_files:
#     print(f"\n{'='*50}")
#     print(f"Processing: {tif_file}")
#     print(f"{'='*50}")
#     
#     tif_path = os.path.join(dat_dir, tif_file)
#     try:
#         ndvi, valid_mask, metadata = process_mcd19a1_file(tif_path)
#     except Exception as e:
#         print(f"Error processing {tif_file}: {str(e)}")
#         continue

# %%