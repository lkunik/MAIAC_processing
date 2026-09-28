#!/usr/bin/env python3
#
# Author: Lewis Kunik - University of Utah
# Contact: lewis.kunik@utah.edu
#

#########################################
# Load packages
#########################################
#%%

import os
import sys
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from rasterio.enums import Resampling
import glob
import geopandas as gpd


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
#%%
#########################################
# Define Global Filepaths
#########################################


base_dir = Path('/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/01d_rect/std_anom/QCfilt_CloudFree')
output_dir = Path('/uufs/chpc.utah.edu/common/home/lin-group23/ltk/MODIS/MCD19A1_analysis/by_ecoregion')
output_dir.mkdir(parents=True, exist_ok=True)

fire_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/MODIS/BurnArea_MCD64A1/01d/NAm_running10yrSum'
burnarea_threshold = 0.1

#%%
# Input files
SPEI01_file = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/SPEIbase/raw/spei01.nc'
SPEI12_file = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/SPEIbase/raw/spei12.nc'
ENF_mask_file = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/ESA_CCI/LandCover/needleleaf_evergreen_open_closed_forest_fraction_0.01deg_north_america.nc'
ENF_threshold = 0.99


EPA_ecoregion_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/EPA/ecoregions/'
EPA_ecoregion_L1_file = os.path.join(EPA_ecoregion_dir, 'L1', 'NA_CEC_Eco_Level1_EPSG4326_regrid.shp')
EPA_ecoregion_L2_file = os.path.join(EPA_ecoregion_dir, 'L2/NAm_shp_FOR_MAIAC_PROC', 'NA_CEC_Eco_Level2_withOther.shp')
EPA_ecoregion_L3_file = os.path.join(EPA_ecoregion_dir, 'L3/NAm_shp_FOR_MAIAC_PROC', 'NAm_cec_eco_l3_v2.shp')


########################################
# Define Global Variables and constants
#########################################

timing_abbrev = 'JJA'


ecoregions_l3_list = [
'Thompson-Okanogan Plateau',
'Central Basin and Range',
'Colorado Plateaus',
'Arizona/New Mexico Mountains',
'Sierra Madre Occidental with Conifer, Oak, and Mixed Forests',
'Interior Forested Lowlands and Uplands',
'Interior Bottomlands',
'Yukon Flats',
'Ogilvie Mountains',
'Mackenzie and Selwyn Mountains',
'Peel River and Nahanni Plateaus',
'Great Bear Plains',
'Hay and Slave River Lowlands',
'Kazan River and Selwyn Lake Uplands',
'La Grande Hills and New Quebec Central Plateau',
'Smallwood Uplands',
'Ungava Bay Basin and George Plateau',
'Coppermine River and Tazin Lake Uplands',
'Hudson Bay and James Bay Lowlands',
'Athabasca Plain and Churchill River Upland',
'Lake Nipigon and Lac Seul Upland',
'Central Laurentians and Mecatina Plateau',
'Hayes River Upland and Big Trout Lake',
'Abitibi Plains and Riviere Rupert Plateau',
'Algonquin/Southern Laurentians',
'Northern Appalachian and Atlantic Maritime Highlands',
'Mid-Boreal Uplands and Peace-Wabaska Lowlands',
'Clear Hills and Western Alberta Upland',
'Mid-Boreal Lowland and Interlake Plain',
'Interior Highlands and Klondike Plateau',
'Copper Plateau',
'Watson Highlands',
'Yukon-Stikine Highlands/Boreal Mountains and Plateaus',
'Skeena-Omineca-Central Canadian Rocky Mountains',
'Middle Rockies',
'Klamath Mountains',
'Sierra Nevada',
'Wasatch and Uinta Mountains',
'Southern Rockies',
'Idaho Batholith',
'Chilcotin Ranges and Fraser Plateau',
'Columbia Mountains/Northern Rockies',
'Canadian Rockies',
'North Cascades',
'Blue Mountains',
'Coastal Western Hemlock-Sitka Spruce Forests',
'Pacific and Nass Ranges',
'Coast Range',
'Southeastern Plains',
'South Central Plains',
'Southern Coastal Plain',
'Cascades',
'Eastern Cascades Slopes and Foothills'
]

MODIS_tiles = [
"h07v06", 
"h08v04",
"h08v05",
"h08v06",
"h08v07",
"h09v02",
"h09v03", 
"h09v04",
"h09v05",
"h09v06",
"h09v07",
"h09v08", 
"h10v02",
"h10v03",
"h10v04",
"h10v05",
"h10v06",
"h10v07",
"h10v08",
"h11v02",
"h11v03",
"h11v04",
"h11v05",
"h11v06",
"h11v07",
"h12v01",
"h12v02",
"h12v03",
"h12v04",
"h12v05",
"h13v01",
"h13v02",
"h13v03",
"h13v04",
"h14v01",
"h14v02",
"h14v03",
"h14v04",
"h15v01",
"h15v02",
]

annualize_months_dict = {'MJJ':[5, 6, 7],
                    'JJA':[6, 7, 8],
                    'JAS':[7, 8, 9],
                    'ASO':[8, 9, 10]}

annualize_months_abbrs = ['MJJ', 'JJA', 'JAS', 'ASO']
annualize_months_descrs = ['May June July', 'June July August', 'July August September', 'August September October']


#########################################
# Define Global Functions
#########################################



#########################################
# Begin Main
#########################################



# ecoregion = sys.argv[1]  # e.g. 'Arizona~New_Mexico_Mountains'
ecoregion = 'Athabasca_Plain_and_Churchill_River_Upland' #'Arizona~New_Mexico_Mountains'
ecoregion_withSpaceSlash = ecoregion.replace('_', ' ').replace("~", "/").replace(",", "") #Arizona/New Mexico Mountains

ecoregions_gdf = gpd.read_file(EPA_ecoregion_L3_file)


# print(ecoregion_l2_gdf['NA_L2NAME'].unique())
ecoregion_gdf = ecoregions_gdf[ecoregions_gdf.NA_L3NAME.str.replace(",", "") == ecoregion_withSpaceSlash]
ecoregion_geom = ecoregion_gdf.geometry


print("Loading ENF mask...")
ENF_xr = xr.open_dataset(ENF_mask_file, decode_coords='all')['fraction_ENF'].squeeze()

print("Loading SPEI data...")
SPEI01_ds = xr.open_dataset(SPEI01_file, decode_coords='all')
SPEI01_xr = SPEI01_ds['spei']

SPEI12_ds = xr.open_dataset(SPEI12_file, decode_coords='all')
SPEI12_xr = SPEI12_ds['spei']


#%%
# Convert time to datetime and filter for summer months (e.g. JAS: July=7, Aug=8, Sept=9)
summer_mask = SPEI01_xr.time.dt.month.isin(annualize_months_dict[timing_abbrev])
SPEI01_summer = SPEI01_xr.isel(time=summer_mask)

summer_mask = SPEI12_xr.time.dt.month.isin(annualize_months_dict[timing_abbrev])
SPEI12_summer = SPEI12_xr.isel(time=summer_mask)


# Group by year and take mean for summer window
print("Calculating summer annual means for SPEI...")
SPEI01_summer_annual = SPEI01_summer.groupby(SPEI01_summer.time.dt.year).mean('time')
SPEI01_summer_annual = SPEI01_summer_annual.rename({'year': 'year'})

SPEI12_summer_annual = SPEI12_summer.groupby(SPEI12_summer.time.dt.year).mean('time')
SPEI12_summer_annual = SPEI12_summer_annual.rename({'year': 'year'})

output_file = output_dir / f'MCD19_SPEI_values_{timing_abbrev}_{ecoregion}.parquet'

# Process each tile
all_data = []

# tile = MODIS_tiles[0]
for tile in MODIS_tiles:
    print(f"\n{'='*60}")
    print(f"Processing tile {tile}...")
    print(f"{'='*60}")

    # Find the annual summer file for this tile
    tile_files = list(base_dir.glob(f'{tile}/annualized/*annual_{timing_abbrev}*.nc'))
    if not tile_files:
        print(f"  WARNING: No file found for {tile}, skipping")
        continue

    tile_file = tile_files[0]
    print(f"  Loading: {tile_file.name}")

    # Load MODIS data
    MCD19_xr_raw = xr.open_dataset(tile_file, decode_coords='all')
    MCD19_xr_raw.rio.write_crs("EPSG:4326", inplace=True)

    MCD19_xr_xmin = np.min(MCD19_xr_raw.x.values)
    MCD19_xr_xmax = np.max(MCD19_xr_raw.x.values)
    MCD19_xr_ymin = np.min(MCD19_xr_raw.y.values)
    MCD19_xr_ymax = np.max(MCD19_xr_raw.y.values)

    ecoregion_bounds = ecoregion_geom.total_bounds  # [minx, miny, maxx, maxy]
    eco_xmin, eco_ymin, eco_xmax, eco_ymax = ecoregion_bounds

    # Check if ecoregion is outside of tile bounds - if so, skip
    if eco_xmin > MCD19_xr_xmax or eco_xmax < MCD19_xr_xmin or eco_ymin > MCD19_xr_ymax or eco_ymax < MCD19_xr_ymin:
        print(f"  Ecoregion is outside of tile bounds, skipping tile {tile}")
        continue

    MCD19_xr = MCD19_xr_raw.rio.clip(ecoregion_geom, drop=False)

    # if (MCD19_xr_clip.sel(year=2020)['Sur_refl1'] > 0).sum().values <= 100:
    #     print(f"  skipping tile, Only {(MCD19_xr_clip['Sur_refl1'] > 0).sum().values} valid pixels in clipped tile, skipping")
    #     continue
    
    # Regrid ENF mask to this tile's grid
    print("  Regridding ENF mask to tile grid...")
    ENF_regrid = ENF_xr.rio.write_crs("EPSG:4326").rio.reproject_match(
        MCD19_xr, resampling=Resampling.bilinear, nodata=np.nan
    )

    # Apply ENF threshold to create mask
    ENF_mask = ENF_regrid >= ENF_threshold
    n_valid_pixels = ENF_mask.sum().values
    print(f"  Valid ENF pixels in tile: {n_valid_pixels}")

    if n_valid_pixels == 0:
        print(f"  No valid pixels, skipping tile")
        MCD19_xr.close()
        continue

    # Regrid SPEI to this tile's grid
    print("  Regridding SPEI to tile grid...")
    SPEI01_regrid = SPEI01_summer_annual.rio.write_crs("EPSG:4326").rio.reproject_match(
        MCD19_xr, resampling=Resampling.bilinear, nodata=np.nan
    )
    SPEI12_regrid = SPEI12_summer_annual.rio.write_crs("EPSG:4326").rio.reproject_match(
        MCD19_xr, resampling=Resampling.bilinear, nodata=np.nan
    )
    # Create coordinate meshgrid once
    lon_2d, lat_2d = np.meshgrid(MCD19_xr.x.values, MCD19_xr.y.values)

    # Process each year
    years_available = MCD19_xr.year.values
    print(f"  Processing {len(years_available)} years: {years_available[0]}-{years_available[-1]}")

    for year in years_available:
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
        else:
            print(f"    WARNING: Burn mask file not found: {burn_mask_file}")
            burn_portion_2d = np.zeros_like(NDVI_year)

        # Burn mask = True where NO BURN
        # Burn mask = False where BURN DETECTED IN PAST 10 YEARS
        burn_mask_2d = burn_portion_2d < burnarea_threshold

        # Get SPEI for this year if available
        if year in SPEI12_regrid.year.values:
            SPEI01_year = SPEI01_regrid.sel(year=year).values
            SPEI12_year = SPEI12_regrid.sel(year=year).values
        else:
            print(f"    WARNING: SPEI not available for year {year}, filling with NaN")
            SPEI01_year = np.full_like(NDVI_year, np.nan)
            SPEI12_year = np.full_like(NDVI_year, np.nan)
        
        # Apply ENF mask to all variables
        ENF_mask_2d = ENF_mask.values
        
        # Only keep pixels where ENF mask is valid and all data present
        valid_mask = (ENF_mask_2d & 
                      burn_mask_2d &
                     ~np.isnan(NDVI_year) & 
                     ~np.isnan(SPEI01_year) & 
                     ~np.isnan(SPEI12_year))
        n_valid = valid_mask.sum()
        
        if n_valid == 0:
            continue
        
        # Extract valid pixels
        df_year = pd.DataFrame({
            'year': year,
            'tile': tile,
            'lon': lon_2d[valid_mask],
            'lat': lat_2d[valid_mask],
            'NDVI_std_anomaly': NDVI_year[valid_mask],
            'CCI_std_anomaly': CCI_year[valid_mask],
            'NIRv_std_anomaly': NIRv_year[valid_mask],
            'NDMI_std_anomaly': NDMI_year[valid_mask],
            'SPEI01': SPEI01_year[valid_mask],
            'SPEI12': SPEI12_year[valid_mask],
        })
        
        all_data.append(df_year)
        print(f"    Year {year}: {n_valid} valid pixels")

    MCD19_xr.close()
    print(f"  Tile {tile} complete")

if len(all_data) == 0:
    print("No valid data extracted for any tile/year combination. Exiting.")
    sys.exit(0)

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
print(f"\nVariable ranges:")
print(f"  SPEI01: {df_all['SPEI01'].min():.2f} to {df_all['SPEI01'].max():.2f}")
print(f"  SPEI12: {df_all['SPEI12'].min():.2f} to {df_all['SPEI12'].max():.2f}")
print(f"\nFile size: {output_file.stat().st_size / 1e6:.1f} MB")
print(f"Output file: {output_file}")
# %%
