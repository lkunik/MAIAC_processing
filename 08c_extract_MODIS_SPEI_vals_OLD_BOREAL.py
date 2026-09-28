#!/usr/bin/env python3
#
# Author: Lewis Kunik - University of Utah
# Contact: lewis.kunik@utah.edu
#

#%%

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from rasterio.enums import Resampling
import glob

#%%
# Configuration
MODIS_tiles = ["h12v04", "h09v03", "h10v03", "h12v03", "h13v03", "h14v03", "h12v02", "h10v02"]
base_dir = Path('/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/01d_rect/QCfilt_CloudFree')
output_dir = Path('/uufs/chpc.utah.edu/common/home/lin-group23/ltk/MODIS/MCD19A1_analysis')
output_dir.mkdir(parents=True, exist_ok=True)

#%%
# Input files
SPEI_file = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/SPEIbase/raw/spei12.nc'
ENF_mask_file = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/ESA_CCI/LandCover/needleleaf_evergreen_closed_forest_fraction_0.01deg_north_america.nc'
ENF_threshold = 0.7

print("Loading ENF mask...")
ENF_xr = xr.open_dataset(ENF_mask_file, decode_coords='all')['fraction_ENF'].squeeze()

print("Loading SPEI data...")
SPEI_ds = xr.open_dataset(SPEI_file, decode_coords='all')
SPEI_xr = SPEI_ds['spei']

#%%
# Convert time to datetime and filter for JAS months (July=7, Aug=8, Sept=9)
JAS_mask = SPEI_xr.time.dt.month.isin([7, 8, 9])
SPEI_JAS = SPEI_xr.isel(time=JAS_mask)

# Group by year and take mean for JAS window
print("Calculating JAS annual means for SPEI...")
SPEI_JAS_annual = SPEI_JAS.groupby(SPEI_JAS.time.dt.year).mean('time')
SPEI_JAS_annual = SPEI_JAS_annual.rename({'year': 'year'})

output_file = output_dir / 'MODIS_SPEI12_values_JAS.parquet'

# Process each tile
all_data = []

# tile = MODIS_tiles[0]
for tile in MODIS_tiles:
    print(f"\n{'='*60}")
    print(f"Processing tile {tile}...")
    print(f"{'='*60}")

    # Find the annual JAS file for this tile
    tile_files = list(base_dir.glob(f'{tile}/annualized/*annual_JAS*.nc'))
    if not tile_files:
        print(f"  WARNING: No file found for {tile}, skipping")
        # continue

    tile_file = tile_files[0]
    print(f"  Loading: {tile_file.name}")

    # Load MODIS data
    MCD19 = xr.open_dataset(tile_file, decode_coords='all')

    # Regrid ENF mask to this tile's grid
    print("  Regridding ENF mask to tile grid...")
    ENF_regrid = ENF_xr.rio.write_crs("EPSG:4326").rio.reproject_match(
        MCD19, resampling=Resampling.bilinear, nodata=np.nan
    )

    # Apply ENF threshold to create mask
    ENF_mask = ENF_regrid >= ENF_threshold
    n_valid_pixels = ENF_mask.sum().values
    print(f"  Valid ENF pixels in tile: {n_valid_pixels}")

    if n_valid_pixels == 0:
        print(f"  No valid pixels, skipping tile")
        MCD19.close()
        # continue

    # Regrid SPEI to this tile's grid
    print("  Regridding SPEI to tile grid...")
    SPEI_regrid = SPEI_JAS_annual.rio.write_crs("EPSG:4326").rio.reproject_match(
        MCD19, resampling=Resampling.bilinear, nodata=np.nan
    )

    # Create coordinate meshgrid once
    lon_2d, lat_2d = np.meshgrid(MCD19.x.values, MCD19.y.values)

    # Process each year
    years_available = MCD19.year.values
    print(f"  Processing {len(years_available)} years: {years_available[0]}-{years_available[-1]}")

    for year in years_available:
        # Extract data for this year
        NDVI_year = MCD19['NDVI'].sel(year=year).values
        CCI_year = MCD19['CCI'].sel(year=year).values
        NIRv_year = MCD19['NIRv'].sel(year=year).values
        NDMI_year = MCD19['NDMI'].sel(year=year).values

        # Get SPEI for this year if available
        if year in SPEI_regrid.year.values:
            SPEI_year = SPEI_regrid.sel(year=year).values
        else:
            print(f"    WARNING: SPEI not available for year {year}, filling with NaN")
            SPEI_year = np.full_like(NDVI_year, np.nan)
        
        # Apply ENF mask to all variables
        ENF_mask_2d = ENF_mask.values
        
        # Only keep pixels where ENF mask is valid
        valid_mask = ENF_mask_2d & ~np.isnan(NDVI_year) & ~np.isnan(SPEI_year)
        n_valid = valid_mask.sum()
        
        # if n_valid == 0:
        #     continue
        
        # Extract valid pixels
        df_year = pd.DataFrame({
            'year': year,
            'tile': tile,
            'lon': lon_2d[valid_mask],
            'lat': lat_2d[valid_mask],
            'NDVI': NDVI_year[valid_mask],
            'CCI': CCI_year[valid_mask],
            'NIRv': NIRv_year[valid_mask],
            'NDMI': NDMI_year[valid_mask],
            'SPEI': SPEI_year[valid_mask],
        })
        
        all_data.append(df_year)
        print(f"    Year {year}: {n_valid} valid pixels")

    MCD19.close()
    print(f"  Tile {tile} complete")

# Combine all data
print(f"\n{'='*60}")
print("Combining all tiles...")
df_all = pd.concat(all_data, ignore_index=True)

# Save to parquet
print(f"Saving to {output_file}...")
df_all.to_parquet(output_file, index=False, compression='snappy')

print(f"\n{'='*60}")
print("EXTRACTION COMPLETE")
print(f"{'='*60}")
print(f"Total valid pixel-years: {len(df_all):,}")
print(f"Years range: {df_all['year'].min():.0f} - {df_all['year'].max():.0f}")
print(f"Tiles processed: {df_all['tile'].nunique()}")
print(f"File size: {output_file.stat().st_size / 1e6:.1f} MB")
print(f"Output file: {output_file}")
# %%
