#!/usr/bin/env python3
# extract_modis_terraclimate_values.py

#%%
import os
import sys
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from rasterio.enums import Resampling

# plotting packages
import matplotlib
from matplotlib import pyplot as plt  # primary plotting module
import matplotlib.colors as mcolors

# Custom packages
utils_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/pyutil'
if utils_dir not in sys.path:
    sys.path.append(utils_dir)
from MODIS_tile_corners import MODIS_tilecorners_df
import data_utils


import geopandas as gpd
from shapely.geometry import mapping
import cartopy.io.shapereader as shpreader
import cartopy.feature as cfeature
from shapely.geometry import Point
from shapely.geometry import Polygon
from pyproj import CRS

import cartopy.crs as ccrs  # spatial plotting library 
from cartopy.io import img_tiles  # cartopy's implementation of webtiles
from matplotlib.ticker import FuncFormatter as FFmt  # need this for formatting plot ticks
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.path import Path as mplPath
from matplotlib.collections import PatchCollection




#%%
# Configuration
MODIS_tiles = ["h12v04", "h09v03", "h10v03", "h12v03", "h13v03", "h14v03", "h12v02", "h10v02", "h11v02", "h11v03"]  # List of MODIS tiles to process

base_dir = Path('/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/01d_rect/std_anom/QCfilt_CloudFree')
output_dir = Path('/uufs/chpc.utah.edu/common/home/lin-group23/ltk/MODIS/MCD19A1_analysis')
output_dir.mkdir(parents=True, exist_ok=True)

fire_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/MODIS/BurnArea_MCD64A1/01d/NAm_Boreal_running10yrSum'
burnarea_threshold = 0.1

# Input files
terraclimate_dir = Path('/uufs/chpc.utah.edu/common/home/lin-group23/ltk/TerraClimate')
ENF_mask_file = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/ESA_CCI/LandCover/needleleaf_evergreen_closed_forest_fraction_0.01deg_north_america.nc'
ENF_threshold = 0.8

timing_abbrev = 'JJA'
plot_basedir = f'/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/plots/drought/scatter_annual/QCfilt_CloudFree/std_anomaly_{timing_abbrev}/QC/'

PDSI_file = terraclimate_dir / f'PDSI/annualized/TerraClimate_PDSI_NAm_annual_{timing_abbrev}.nc'
CWD_file = terraclimate_dir / f'CWD/annualized/anomaly/TerraClimate_CWD_NAm_annual_Anomaly_{timing_abbrev}.nc'
PET_file = terraclimate_dir / f'PET/annualized/anomaly/TerraClimate_PET_NAm_annual_Anomaly_{timing_abbrev}.nc'
SoilM_file = terraclimate_dir / f'SoilM/annualized/anomaly/TerraClimate_SoilM_NAm_annual_Anomaly_{timing_abbrev}.nc'
VPD_file = terraclimate_dir / f'VPD/annualized/anomaly/TerraClimate_VPD_NAm_annual_Anomaly_{timing_abbrev}.nc'

# province boundaries
provinc_bodr = cfeature.NaturalEarthFeature(category='cultural', 
    name='admin_1_states_provinces_lines', scale='50m', facecolor='none', edgecolor='k')


#%%
print("Loading ENF mask...")
ENF_xr = xr.open_dataset(ENF_mask_file, decode_coords='all')['fraction_ENF'].squeeze()

print("Loading TerraClimate data...")
# Load all TerraClimate variables

PDSI_xr = xr.open_dataset(PDSI_file, decode_coords='all')['PDSI']
CWD_xr = xr.open_dataset(CWD_file, decode_coords='all')['CWD_anomaly']  # Note: variable name is 'def'
PET_xr = xr.open_dataset(PET_file, decode_coords='all')['PET_anomaly']
SoilM_xr = xr.open_dataset(SoilM_file, decode_coords='all')['SoilM_anomaly']
VPD_xr = xr.open_dataset(VPD_file, decode_coords='all')['VPD_anomaly']

print(f"TerraClimate data loaded:")
print(f"  Years: {PDSI_xr.year.values[0]} - {PDSI_xr.year.values[-1]}")
print(f"  PDSI range: {float(PDSI_xr.min().values):.2f} to {float(PDSI_xr.max().values):.2f}")
print(f"  CWD Anomaly range: {float(CWD_xr.min().values):.1f} to {float(CWD_xr.max().values):.1f} mm")
print(f"  PET Anomaly range: {float(PET_xr.min().values):.1f} to {float(PET_xr.max().values):.1f} mm")
print(f"  SoilM Anomaly range: {float(SoilM_xr.min().values):.1f} to {float(SoilM_xr.max().values):.1f} mm")
print(f"  VPD Anomaly range: {float(VPD_xr.min().values):.2f} to {float(VPD_xr.max().values):.2f} kPa")

output_file = output_dir / f'MCD19_TerraClimate_StdAnom_values_{timing_abbrev}.parquet'

# Process each tile
all_data = []


def plot_map(dat_xr, extent, filename, plotTitle, cmap='viridis', cbar_label=None, 
             vmin=None, vmax=None, center0=False):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot the ENF mask to verify regridding worked
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create figure with map projection
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    
    if vmin is None:
        vmin = np.nanquantile(dat_xr.values, 0.01)
    if vmax is None:
        vmax = np.nanquantile(dat_xr.values, 0.99)
    
    if center0:
        abs_max = max(abs(vmin), abs(vmax))
        vmin, vmax = -abs_max, abs_max


    # Plot the data
    if 'x' in dat_xr.coords and 'y' in dat_xr.coords:
        im = ax.pcolormesh(dat_xr.x, dat_xr.y, dat_xr,
                        transform=ccrs.PlateCarree(),
                        cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        im = ax.pcolormesh(dat_xr.lon, dat_xr.lat, dat_xr,
                        transform=ccrs.PlateCarree(),
                        cmap=cmap, vmin=vmin, vmax=vmax)        
    
    # Add map features
    ax.coastlines(resolution='50m', linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor='gray')
    ax.add_feature(provinc_bodr, linewidth=0.3, edgecolor='gray')

    
    # Set extent to match the data region
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02, fraction=0.02)
    if cbar_label:
        cbar.set_label(cbar_label)
    
    # Set title
    ax.set_title(plotTitle)
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    # Save figure
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {plotTitle} map to {filename}')
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

plot_year = 2010
#%%
tile = MODIS_tiles[3] #"h12v04", "h09v03", "h10v03", "h12v03", "h13v03", "h14v03", "h12v02", "h10v02", "h11v02", "h11v03"
# for tile in MODIS_tiles:
print(f"\n{'='*60}")
print(f"Processing tile {tile}...")
print(f"{'='*60}")

plot_dir = os.path.join(plot_basedir, tile)
Path(plot_dir).mkdir(parents=True, exist_ok=True)
tile_corners = MODIS_tilecorners_df.loc[tile]

# Find the annual file for this tile
tile_files = list(base_dir.glob(f'{tile}/annualized/*annual_{timing_abbrev}*.nc'))
if not tile_files:
    print(f"  WARNING: No file found for {tile}, skipping")
    # continue

tile_file = tile_files[0]
print(f"  Loading: {tile_file.name}")

# Load MODIS data
MCD19_xr = xr.open_dataset(tile_file, decode_coords='all')
MCD19_xr_xmin = np.min(MCD19_xr.x.values)
MCD19_xr_xmax = np.max(MCD19_xr.x.values)
MCD19_xr_ymin = np.min(MCD19_xr.y.values)
MCD19_xr_ymax = np.max(MCD19_xr.y.values)

ESA_CCI_xr_tile = ENF_xr.sel(y=slice(MCD19_xr_ymin, MCD19_xr_ymax), x=slice(MCD19_xr_xmin, MCD19_xr_xmax))

assert data_utils.check_if_xr_grids_equal(ESA_CCI_xr_tile, MCD19_xr)



# # Regrid ENF mask to this tile's grid
# print("  Regridding ENF mask to tile grid...")
# ENF_regrid = ESA_CCI_xr_tile.rio.write_crs("EPSG:4326").rio.reproject_match(
#     MCD19_xr, resampling=Resampling.bilinear, nodata=np.nan
# )

filename = os.path.join(plot_dir, f'ESA_CCI_ENF_Mask_{tile}.png')
plot_map(ESA_CCI_xr_tile, extent=[MCD19_xr_xmin, MCD19_xr_xmax, MCD19_xr_ymin, MCD19_xr_ymax], 
filename=filename, plotTitle=f'ESA CCI ENF Mask - tile {tile}', 
cmap='Greens', cbar_label='ENF Mask Value', vmin=0, vmax=1)


# Apply ENF threshold to create mask
ENF_mask = ESA_CCI_xr_tile >= ENF_threshold
n_valid_pixels = ENF_mask.sum().values
print(f"  Valid ENF pixels in tile: {n_valid_pixels}")

if n_valid_pixels == 0:
    print(f"  No valid pixels, skipping tile")
    MCD19_xr.close()
    sys.exit(1)

# Regrid all TerraClimate variables to this tile's grid
print("  Regridding TerraClimate variables to tile grid...")
PDSI_regrid = PDSI_xr.rio.write_crs("EPSG:4326").rio.reproject_match(
    MCD19_xr, resampling=Resampling.bilinear, nodata=np.nan
)
CWD_regrid = CWD_xr.rio.write_crs("EPSG:4326").rio.reproject_match(
    MCD19_xr, resampling=Resampling.bilinear, nodata=np.nan
)
PET_regrid = PET_xr.rio.write_crs("EPSG:4326").rio.reproject_match(
    MCD19_xr, resampling=Resampling.bilinear, nodata=np.nan
)
SoilM_regrid = SoilM_xr.rio.write_crs("EPSG:4326").rio.reproject_match(
    MCD19_xr, resampling=Resampling.bilinear, nodata=np.nan
)
VPD_regrid = VPD_xr.rio.write_crs("EPSG:4326").rio.reproject_match(
    MCD19_xr, resampling=Resampling.bilinear, nodata=np.nan
)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
### PLOT MAPS FOR SANITY CHECK
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
filename = os.path.join(plot_dir, f'PDSI_{str(plot_year)}_{tile}_01d.png')
plot_map(PDSI_regrid.sel(year = plot_year), extent=[MCD19_xr_xmin, MCD19_xr_xmax, MCD19_xr_ymin, MCD19_xr_ymax], 
filename=filename, plotTitle=f'PDSI {plot_year} - tile {tile}', 
cmap='RdBu', cbar_label='PDSI', center0=True)

filename = os.path.join(plot_dir, f'CWD_{str(plot_year)}_{tile}_01d.png')
plot_map(CWD_regrid.sel(year=plot_year), extent=[MCD19_xr_xmin, MCD19_xr_xmax, MCD19_xr_ymin, MCD19_xr_ymax], 
filename=filename, plotTitle=f'CWD {plot_year} - tile {tile}', 
cmap='RdBu_r', cbar_label='CWD Anomaly (mm)', center0=True)

filename = os.path.join(plot_dir, f'PET_{str(plot_year)}_{tile}_01d.png')
plot_map(PET_regrid.sel(year=plot_year), extent=[MCD19_xr_xmin, MCD19_xr_xmax, MCD19_xr_ymin, MCD19_xr_ymax], 
filename=filename, plotTitle=f'PET {plot_year} - tile {tile}', 
cmap='RdBu_r', cbar_label='PET Anomaly (mm)', center0=True)

filename = os.path.join(plot_dir, f'SoilM_{str(plot_year)}_{tile}_01d.png')
plot_map(SoilM_regrid.sel(year=plot_year), extent=[MCD19_xr_xmin, MCD19_xr_xmax, MCD19_xr_ymin, MCD19_xr_ymax], 
filename=filename, plotTitle=f'Soil Moisture {plot_year} - tile {tile}', 
cmap='RdBu', cbar_label='Soil Moisture Anomaly (mm)', center0=True)

filename = os.path.join(plot_dir, f'VPD_{str(plot_year)}_{tile}_01d.png')
plot_map(VPD_regrid.sel(year=plot_year), extent=[MCD19_xr_xmin, MCD19_xr_xmax, MCD19_xr_ymin, MCD19_xr_ymax], 
filename=filename, plotTitle=f'VPD {plot_year} - tile {tile}', 
cmap='RdBu_r', cbar_label='VPD Anomaly (kPa)', center0=True)

# End Sanity check plots
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Create coordinate meshgrid once
lon_2d, lat_2d = np.meshgrid(MCD19_xr.x.values, MCD19_xr.y.values)

# Process each year
years_available = MCD19_xr.year.values
print(f"  Processing {len(years_available)} years: {years_available[0]}-{years_available[-1]}")

# for year in years_available:
year = plot_year
# Extract MODIS data for this year
NDVI_year = MCD19_xr['NDVI_std_anomaly'].sel(year=year).values
CCI_year = MCD19_xr['CCI_std_anomaly'].sel(year=year).values
NIRv_year = MCD19_xr['NIRv_std_anomaly'].sel(year=year).values
NDMI_year = MCD19_xr['NDMI_std_anomaly'].sel(year=year).values

# Load burn area mask for this year
burn_mask_file = os.path.join(fire_dir, f'MCD64A1.061_01d_reproc_10yrsum_{year}.nc')
if os.path.exists(burn_mask_file):
    burn_xr_slice = xr.open_dataset(burn_mask_file).sel(y=slice(MCD19_xr_ymin, MCD19_xr_ymax), x=slice(MCD19_xr_xmin, MCD19_xr_xmax))
    
    if not data_utils.check_if_xr_grids_equal(burn_xr_slice, MCD19_xr):
        # Regrid burn mask to match MCD19 grid, filling with 0 where data unavailable
        burn_xr = burn_xr_slice.reindex(
            y=MCD19_xr.y, x=MCD19_xr.x, fill_value=0
        )
    else:
        burn_xr = burn_xr_slice

    burn_portion_2d = burn_xr['burn_portion'].values
    filename = os.path.join(plot_dir, f'Burn_portion_{str(year)}_{tile}_01d.png')
    plot_map(burn_xr['burn_portion'], extent=[MCD19_xr_xmin, MCD19_xr_xmax, MCD19_xr_ymin, MCD19_xr_ymax], 
    filename=filename, plotTitle=f'Burned area mask {year} - tile {tile}', 
    cmap='Reds', cbar_label='Burned area portion', center0=False, vmin = 0, vmax=1)

else:
    print(f"    WARNING: Burn mask file not found: {burn_mask_file}")
    burn_portion_2d = np.zeros_like(NDVI_year)

# Burn mask = True where NO BURN
# Burn mask = False where BURN DETECTED IN PAST 10 YEARS
burn_mask_2d = burn_portion_2d < burnarea_threshold

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
### PLOT MAPS FOR SANITY CHECK
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

filename = os.path.join(plot_dir, f'CCI_Anom_{str(year)}_{tile}_01d.png')
plot_map(MCD19_xr['CCI_std_anomaly'].sel(year=year), extent=[MCD19_xr_xmin, MCD19_xr_xmax, MCD19_xr_ymin, MCD19_xr_ymax], 
filename=filename, plotTitle=f'CCI Anomaly {year} - tile {tile}', 
cmap='RdBu', cbar_label='CCI Anomaly', center0=True)

filename = os.path.join(plot_dir, f'NDVI_Anom_{str(year)}_{tile}_01d.png')
plot_map(MCD19_xr['NDVI_std_anomaly'].sel(year=year), extent=[MCD19_xr_xmin, MCD19_xr_xmax, MCD19_xr_ymin, MCD19_xr_ymax], 
filename=filename, plotTitle=f'NDVI Anomaly {year} - tile {tile}', 
cmap='RdBu', cbar_label='NDVI Anomaly', center0=True)

filename = os.path.join(plot_dir, f'NIRv_Anom_{str(year)}_{tile}_01d.png')
plot_map(MCD19_xr['NIRv_std_anomaly'].sel(year=year), extent=[MCD19_xr_xmin, MCD19_xr_xmax, MCD19_xr_ymin, MCD19_xr_ymax], 
filename=filename, plotTitle=f'NIRv Anomaly {year} - tile {tile}', 
cmap='RdBu', cbar_label='NIRv Anomaly', center0=True)

filename = os.path.join(plot_dir, f'NDMI_Anom_{str(year)}_{tile}_01d.png')
plot_map(MCD19_xr['NDMI_std_anomaly'].sel(year=year), extent=[MCD19_xr_xmin, MCD19_xr_xmax, MCD19_xr_ymin, MCD19_xr_ymax], 
filename=filename, plotTitle=f'NDMI Anomaly {year} - tile {tile}', 
cmap='RdBu', cbar_label='NDMI Anomaly', center0=True)

# End Sanity check plots
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Get TerraClimate data for this year if available
if year in PDSI_regrid.year.values:
    PDSI_year = PDSI_regrid.sel(year=year).values        
    CWD_year = CWD_regrid.sel(year=year).values
    PET_year = PET_regrid.sel(year=year).values
    SoilM_year = SoilM_regrid.sel(year=year).values
    VPD_year = VPD_regrid.sel(year=year).values
else:
    print(f"    WARNING: TerraClimate data not available for year {year}, filling with NaN")
    PDSI_year = np.full_like(NDVI_year, np.nan)
    CWD_year = np.full_like(NDVI_year, np.nan)
    PET_year = np.full_like(NDVI_year, np.nan)
    SoilM_year = np.full_like(NDVI_year, np.nan)
    VPD_year = np.full_like(NDVI_year, np.nan)

# Apply ENF mask to all variables
ENF_mask_2d = ENF_mask.values

# Only keep pixels where ENF mask is valid and all data present
validENF_mask = (ENF_mask_2d & 
                ~np.isnan(NDVI_year) & 
                ~np.isnan(PDSI_year) & 
                ~np.isnan(CWD_year) &
                ~np.isnan(PET_year) &
                ~np.isnan(SoilM_year) &
                ~np.isnan(VPD_year))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
### PLOT MAPS FOR SANITY CHECK
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

### TerraClimate maps
PDSI_plot = PDSI_regrid.sel(year=year).where(validENF_mask, np.nan, drop = False)
filename = os.path.join(plot_dir, f'PDSI_{str(plot_year)}_ENFmask_{tile}_01d.png')
plot_map(PDSI_plot, extent=[MCD19_xr_xmin, MCD19_xr_xmax, MCD19_xr_ymin, MCD19_xr_ymax], 
filename=filename, plotTitle=f'PDSI {plot_year} (ENF+NoBurn mask) - tile {tile}', 
cmap='RdYlBu', cbar_label='PDSI', center0=True)

CWD_plot = CWD_regrid.sel(year=year).where(validENF_mask, np.nan, drop = False)
filename = os.path.join(plot_dir, f'CWD_{str(plot_year)}_ENFmask_{tile}_01d.png')
plot_map(CWD_plot, extent=[MCD19_xr_xmin, MCD19_xr_xmax, MCD19_xr_ymin, MCD19_xr_ymax], 
filename=filename, plotTitle=f'CWD {plot_year} (ENF+NoBurn mask) - tile {tile}', 
cmap='RdYlBu_r', cbar_label='CWD Anomaly (mm)', center0=True)

PET_plot = PET_regrid.sel(year=year).where(validENF_mask, np.nan, drop = False)
filename = os.path.join(plot_dir, f'PET_{str(plot_year)}_ENFmask_{tile}_01d.png')
plot_map(PET_plot, extent=[MCD19_xr_xmin, MCD19_xr_xmax, MCD19_xr_ymin, MCD19_xr_ymax], 
filename=filename, plotTitle=f'PET {plot_year} (ENF+NoBurn mask) - tile {tile}', 
cmap='RdYlBu_r', cbar_label='PET Anomaly (mm)', center0=True)

SoilM_plot = SoilM_regrid.sel(year=year).where(validENF_mask, np.nan, drop = False)
filename = os.path.join(plot_dir, f'SoilM_{str(plot_year)}_ENFmask_{tile}_01d.png')
plot_map(SoilM_plot, extent=[MCD19_xr_xmin, MCD19_xr_xmax, MCD19_xr_ymin, MCD19_xr_ymax], 
filename=filename, plotTitle=f'Soil Moisture {plot_year} (ENF+NoBurn mask) - tile {tile}', 
cmap='RdYlBu', cbar_label='Soil Moisture Anomaly (mm)', center0=True)

VPD_plot = VPD_regrid.sel(year=year).where(validENF_mask, np.nan, drop = False)
filename = os.path.join(plot_dir, f'VPD_{str(plot_year)}_ENFmask_{tile}_01d.png')
plot_map(VPD_plot, extent=[MCD19_xr_xmin, MCD19_xr_xmax, MCD19_xr_ymin, MCD19_xr_ymax], 
filename=filename, plotTitle=f'VPD {plot_year} (ENF+NoBurn mask) - tile {tile}', 
cmap='RdYlBu_r', cbar_label='VPD Anomaly (kPa)', center0=True)

# MCD19 maps
CCI_plot = MCD19_xr['CCI_std_anomaly'].sel(year=year).where(validENF_mask, np.nan, drop = False)
filename = os.path.join(plot_dir, f'CCI_Anom_{str(year)}_ENFmask_{tile}_01d.png')
plot_map(CCI_plot, extent=[MCD19_xr_xmin, MCD19_xr_xmax, MCD19_xr_ymin, MCD19_xr_ymax], 
filename=filename, plotTitle=f'CCI Anomaly {year} (ENF+NoBurn mask) - tile {tile}', 
cmap='RdYlBu', cbar_label='CCI Anomaly', center0=True)

NDVI_plot = MCD19_xr['NDVI_std_anomaly'].sel(year=year).where(validENF_mask, np.nan, drop = False)
filename = os.path.join(plot_dir, f'NDVI_Anom_{str(year)}_ENFmask_{tile}_01d.png')
plot_map(NDVI_plot, extent=[MCD19_xr_xmin, MCD19_xr_xmax, MCD19_xr_ymin, MCD19_xr_ymax], 
filename=filename, plotTitle=f'NDVI Anomaly {year} (ENF+NoBurn mask) - tile {tile}', 
cmap='RdYlBu', cbar_label='NDVI Anomaly', center0=True)

NIRv_plot = MCD19_xr['NIRv_std_anomaly'].sel(year=year).where(validENF_mask, np.nan, drop = False)
filename = os.path.join(plot_dir, f'NIRv_Anom_{str(year)}_ENFmask_{tile}_01d.png')
plot_map(NIRv_plot, extent=[MCD19_xr_xmin, MCD19_xr_xmax, MCD19_xr_ymin, MCD19_xr_ymax], 
filename=filename, plotTitle=f'NIRv Anomaly {year} (ENF+NoBurn mask) - tile {tile}', 
cmap='RdYlBu', cbar_label='NIRv Anomaly', center0=True)

NDMI_plot = MCD19_xr['NDMI_std_anomaly'].sel(year=year).where(validENF_mask, np.nan, drop = False)
filename = os.path.join(plot_dir, f'NDMI_Anom_{str(year)}_ENFmask_{tile}_01d.png')
plot_map(NDMI_plot, extent=[MCD19_xr_xmin, MCD19_xr_xmax, MCD19_xr_ymin, MCD19_xr_ymax], 
filename=filename, plotTitle=f'NDMI Anomaly {year} (ENF+NoBurn mask) - tile {tile}', 
cmap='RdYlBu_r', cbar_label='NDMI Anomaly', center0=True)

# End Sanity check plots
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Only keep pixels where ENF mask is valid and all data present
valid_mask = (ENF_mask_2d & 
                burn_mask_2d &
                ~np.isnan(NDVI_year) & 
                ~np.isnan(PDSI_year) & 
                ~np.isnan(CWD_year) &
                ~np.isnan(PET_year) &
                ~np.isnan(SoilM_year) &
                ~np.isnan(VPD_year))
n_valid = valid_mask.sum()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
### PLOT MAPS FOR SANITY CHECK
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

### TerraClimate maps
PDSI_plot = PDSI_regrid.sel(year=year).where(valid_mask, np.nan, drop = False)
filename = os.path.join(plot_dir, f'PDSI_{str(plot_year)}_ENFnoBurnMask_{tile}_01d.png')
plot_map(PDSI_plot, extent=[MCD19_xr_xmin, MCD19_xr_xmax, MCD19_xr_ymin, MCD19_xr_ymax], 
filename=filename, plotTitle=f'PDSI {plot_year} (ENF+NoBurn mask) - tile {tile}', 
cmap='RdYlBu', cbar_label='PDSI', center0=True)

CWD_plot = CWD_regrid.sel(year=year).where(valid_mask, np.nan, drop = False)
filename = os.path.join(plot_dir, f'CWD_{str(plot_year)}_ENFnoBurnMask_{tile}_01d.png')
plot_map(CWD_plot, extent=[MCD19_xr_xmin, MCD19_xr_xmax, MCD19_xr_ymin, MCD19_xr_ymax], 
filename=filename, plotTitle=f'CWD {plot_year} (ENF+NoBurn mask) - tile {tile}', 
cmap='RdYlBu_r', cbar_label='CWD Anomaly (mm)', center0=True)

PET_plot = PET_regrid.sel(year=year).where(valid_mask, np.nan, drop = False)
filename = os.path.join(plot_dir, f'PET_{str(plot_year)}_ENFnoBurnMask_{tile}_01d.png')
plot_map(PET_plot, extent=[MCD19_xr_xmin, MCD19_xr_xmax, MCD19_xr_ymin, MCD19_xr_ymax], 
filename=filename, plotTitle=f'PET {plot_year} (ENF+NoBurn mask) - tile {tile}', 
cmap='RdYlBu_r', cbar_label='PET Anomaly (mm)', center0=True)

SoilM_plot = SoilM_regrid.sel(year=year).where(valid_mask, np.nan, drop = False)
filename = os.path.join(plot_dir, f'SoilM_{str(plot_year)}_ENFnoBurnMask_{tile}_01d.png')
plot_map(SoilM_plot, extent=[MCD19_xr_xmin, MCD19_xr_xmax, MCD19_xr_ymin, MCD19_xr_ymax], 
filename=filename, plotTitle=f'Soil Moisture {plot_year} (ENF+NoBurn mask) - tile {tile}', 
cmap='RdYlBu', cbar_label='Soil Moisture Anomaly (mm)', center0=True)

VPD_plot = VPD_regrid.sel(year=year).where(valid_mask, np.nan, drop = False)
filename = os.path.join(plot_dir, f'VPD_{str(plot_year)}_ENFnoBurnMask_{tile}_01d.png')
plot_map(VPD_plot, extent=[MCD19_xr_xmin, MCD19_xr_xmax, MCD19_xr_ymin, MCD19_xr_ymax], 
filename=filename, plotTitle=f'VPD {plot_year} (ENF+NoBurn mask) - tile {tile}', 
cmap='RdYlBu_r', cbar_label='VPD Anomaly (kPa)', center0=True)

# MCD19 maps
CCI_plot = MCD19_xr['CCI_std_anomaly'].sel(year=year).where(valid_mask, np.nan, drop = False)
filename = os.path.join(plot_dir, f'CCI_Anom_{str(year)}_ENFnoBurnMask_{tile}_01d.png')
plot_map(CCI_plot, extent=[MCD19_xr_xmin, MCD19_xr_xmax, MCD19_xr_ymin, MCD19_xr_ymax], 
filename=filename, plotTitle=f'CCI Anomaly {year} (ENF+NoBurn mask) - tile {tile}', 
cmap='RdYlBu', cbar_label='CCI Anomaly', center0=True)

NDVI_plot = MCD19_xr['NDVI_std_anomaly'].sel(year=year).where(valid_mask, np.nan, drop = False)
filename = os.path.join(plot_dir, f'NDVI_Anom_{str(year)}_ENFnoBurnMask_{tile}_01d.png')
plot_map(NDVI_plot, extent=[MCD19_xr_xmin, MCD19_xr_xmax, MCD19_xr_ymin, MCD19_xr_ymax], 
filename=filename, plotTitle=f'NDVI Anomaly {year} (ENF+NoBurn mask) - tile {tile}', 
cmap='RdYlBu', cbar_label='NDVI Anomaly', center0=True)

NIRv_plot = MCD19_xr['NIRv_std_anomaly'].sel(year=year).where(valid_mask, np.nan, drop = False)
filename = os.path.join(plot_dir, f'NIRv_Anom_{str(year)}_ENFnoBurnMask_{tile}_01d.png')
plot_map(NIRv_plot, extent=[MCD19_xr_xmin, MCD19_xr_xmax, MCD19_xr_ymin, MCD19_xr_ymax], 
filename=filename, plotTitle=f'NIRv Anomaly {year} (ENF+NoBurn mask) - tile {tile}', 
cmap='RdYlBu', cbar_label='NIRv Anomaly', center0=True)

NDMI_plot = MCD19_xr['NDMI_std_anomaly'].sel(year=year).where(valid_mask, np.nan, drop = False)
filename = os.path.join(plot_dir, f'NDMI_Anom_{str(year)}_ENFnoBurnMask_{tile}_01d.png')
plot_map(NDMI_plot, extent=[MCD19_xr_xmin, MCD19_xr_xmax, MCD19_xr_ymin, MCD19_xr_ymax], 
filename=filename, plotTitle=f'NDMI Anomaly {year} (ENF+NoBurn mask) - tile {tile}', 
cmap='RdYlBu_r', cbar_label='NDMI Anomaly', center0=True)

# End Sanity check plots
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# if n_valid == 0:
#     continue

# # Extract valid pixels
# df_year = pd.DataFrame({
#     'year': year,
#     'tile': tile,
#     'lon': lon_2d[valid_mask],
#     'lat': lat_2d[valid_mask],
#     'NDVI_std_anomaly': NDVI_year[valid_mask],
#     'CCI_std_anomaly': CCI_year[valid_mask],
#     'NIRv_std_anomaly': NIRv_year[valid_mask],
#     'NDMI_std_anomaly': NDMI_year[valid_mask],
#     'PDSI': PDSI_year[valid_mask],
#     'CWD': CWD_year[valid_mask],
#     'PET': PET_year[valid_mask],
#     'SoilM': SoilM_year[valid_mask],
#     'VPD': VPD_year[valid_mask],
# })
    
#     # all_data.append(df_year)
#     # print(f"    Year {year}: {n_valid} valid pixels")

# MCD19_xr.close()
# print(f"  Tile {tile} complete")

# # Combine all data
# print(f"\n{'='*60}")
# print("Combining all tiles...")
# df_all = pd.concat(all_data, ignore_index=True)

# # Save to parquet
# print(f"Saving to {output_file}...")
# df_all.to_parquet(output_file, index=False, compression='snappy')

# print(f"\n{'='*60}")
# print("EXTRACTION COMPLETE")
# print(f"{'='*60}")
# print(f"Total valid pixel-years: {len(df_all):,}")
# print(f"Years range: {df_all['year'].min():.0f} - {df_all['year'].max():.0f}")
# print(f"Tiles processed: {df_all['tile'].nunique()}")
# print(f"\nVariable ranges:")
# print(f"  PDSI: {df_all['PDSI'].min():.2f} to {df_all['PDSI'].max():.2f}")
# print(f"  CWD: {df_all['CWD'].min():.1f} to {df_all['CWD'].max():.1f} mm")
# print(f"  PET: {df_all['PET'].min():.1f} to {df_all['PET'].max():.1f} mm")
# print(f"  SoilM: {df_all['SoilM'].min():.1f} to {df_all['SoilM'].max():.1f} mm")
# print(f"  VPD: {df_all['VPD'].min():.2f} to {df_all['VPD'].max():.2f} kPa")
# print(f"\nFile size: {output_file.stat().st_size / 1e6:.1f} MB")
# print(f"Output file: {output_file}")