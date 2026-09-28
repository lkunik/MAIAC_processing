#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Lewis Kunik - University of Utah
# Contact: lewis.kunik@utah.edu
#
# Step 1 of 2 for the MCD19-vs-MOD13 gridded NDVI cross-comparison (ESSD manuscript, Sec 3.2).
#
# Purpose: for every MCD19 tile / year / 16-day composite period, extract NDVI values at
# pixels falling within each of 4 land-cover-class masks (ENF, DBF/Mixed forest, Grassland,
# Shrubland; fraction > 0.99), and write them to a partitioned Parquet dataset keyed by
# (lat, lon) at 3-decimal precision, plus year and composite_period. This output is read back
# in by the companion script (07b_extract_MOD13_ndvi_and_join.py), which looks up matching
# MOD13 NDVI values by (lat, lon, year, composite_period) and produces the final paired dataset.
#
# Decisions carried over from conversation with Lewis (2026-09-02), recorded here for
# traceability into the manuscript methods section:
#   - No ecoregion-shapefile stratification/clipping is applied in this comparison. The
#     manuscript's planned Fig. 3.2 stratifies only by land-cover/ecosystem type, pooled
#     across the full North American domain, so ecoregion shapefiles are NOT used here.
#   - Land cover stratification uses 4 ESA-CCI-derived fraction masks (paths below), each
#     thresholded at fraction > 0.99 (mutually near-exclusive by construction, since class
#     fractions are bounded and sum <= 1).
#   - MCD19 and MOD13 grids are assumed to be exactly co-registered (same 0.01-deg phase,
#     confirmed via matching x-origin and an integer-pixel y-origin offset in the GeoTransform
#     metadata of the sample files inspected). Matching between the two products is therefore
#     done by exact coordinate value after rounding lat/lon to 3 decimals (per Lewis's
#     instruction), rather than by nearest-neighbor search or regridding.
#   - NDVI values in both MCD19 and MOD13 files are already in physical units; no additional
#     scale factor is applied.
#

#########################################
# Load packages
#########################################
#%%
import os
import re
import glob
import gc
from pathlib import Path

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import xarray as xr
import rioxarray as rxr  # noqa: F401  (registers the .rio accessor)
import pandas as pd
from rasterio.enums import Resampling

import pyarrow as pa
import pyarrow.dataset as pads

import time
from datetime import datetime as dt

#%%
#########################################
# Config
#########################################

QC_option = 'CloudFree_LowAOD_ClearAdj'
MCD19_basedir = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/01d_rect/QCfilt_{QC_option}/'

# Land cover fraction masks, one per ecosystem-type panel in the manuscript figure.
# NOTE: I only know the internal variable name for the ENF-style masks from the seasonality
# script ('fraction_ENF'). For the other 3 files I auto-detect the single data variable at
# load time (see load_lc_fraction below) rather than guessing a name - if a file has more
# than one data variable this will raise an error and you'll need to tell me which to use.
LC_MASKS = {
    'ENF': '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/ESA_CCI/LandCover/needleleaf_evergreen_open_closed_forest_fraction_0.01deg_north_america.nc',
    'DBF_MixedForest': '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/ESA_CCI/LandCover/mixed_forest_fraction_0.01deg_north_america.nc',
    'Grassland': '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/ESA_CCI/LandCover/grassland_fraction_0.01deg_north_america.nc',
    'Shrubland': '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/ESA_CCI/LandCover/shrubland_fraction_0.01deg_north_america.nc',
}
LC_THRESHOLD = 0.99  # fraction > this value counts as "in" that land cover class

ROUND_DECIMALS = 3  # lat/lon rounding for cross-product coordinate matching

OUTPUT_ROOT = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/data/MCD19_MOD13_comparison'
MCD19_PARQUET_DIR = os.path.join(OUTPUT_ROOT, 'mcd19_ndvi_by_landcover')
Path(MCD19_PARQUET_DIR).mkdir(parents=True, exist_ok=True)

# Explicit partition schema (rather than relying on pyarrow's hive-partitioning type
# inference on read-back, which is version-dependent for integer columns) - this MUST match
# the schema used when reading this dataset back in 07b_extract_MOD13_ndvi_and_join.py.
MCD19_PARTITIONING = pads.partitioning(
    pa.schema([("year", pa.int32()), ("composite_period", pa.int32())]),
    flavor="hive",
)

#########################################
# Helper functions
#########################################


def get_all_tiles(base_dir):
    """Find all MODIS tile subdirectories (e.g. h08v04) present under base_dir."""
    tile_re = re.compile(r'^h\d{2}v\d{2}$')
    tiles = sorted(
        d for d in os.listdir(base_dir)
        if tile_re.match(d) and os.path.isdir(os.path.join(base_dir, d))
    )
    return tiles


FILENAME_RE = re.compile(
    r'_(?P<year>\d{4})-(?P<period>\d{2})_rect\.nc$'
)


def parse_year_period(filename):
    """Extract (year, composite_period) as ints from an MCD19 rectified filename."""
    m = FILENAME_RE.search(os.path.basename(filename))
    if m is None:
        return None, None
    return int(m.group('year')), int(m.group('period'))


def load_lc_fraction(path):
    """Open a land-cover fraction NetCDF and return the single data variable as a DataArray."""
    ds = xr.open_dataset(path, decode_coords='all')
    data_vars = list(ds.data_vars)
    if len(data_vars) != 1:
        raise ValueError(
            f"Expected exactly one data variable in {path}, found {data_vars}. "
            "Please specify which variable to use."
        )
    da = ds[data_vars[0]].squeeze()
    return da


def extract_valid_pixels(ndvi_2d, mask_2d, lat_1d, lon_1d, threshold):
    """
    Vectorized extraction of (lat, lon, ndvi) for pixels where mask_2d > threshold
    and ndvi_2d is not NaN. ndvi_2d and mask_2d must share shape (ny, nx); lat_1d has
    length ny, lon_1d has length nx.
    Returns a DataFrame with columns ['lat', 'lon', 'ndvi'], or None if no valid pixels.
    """
    valid = (mask_2d > threshold) & ~np.isnan(ndvi_2d)
    if not np.any(valid):
        return None
    rows, cols = np.where(valid)
    df = pd.DataFrame({
        'lat': np.round(lat_1d[rows], ROUND_DECIMALS),
        'lon': np.round(lon_1d[cols], ROUND_DECIMALS),
        'ndvi': ndvi_2d[rows, cols],
    })
    return df


#########################################
# Begin Main
#########################################
#%%

start_time = time.time()

print("Loading land cover fraction masks (native grid, un-regridded)...")
lc_fraction_native = {lc: load_lc_fraction(path) for lc, path in LC_MASKS.items()}

all_tiles = get_all_tiles(MCD19_basedir)
print(f"Found {len(all_tiles)} MCD19 tiles: {all_tiles}")

# Cache of land-cover masks regridded to each tile's grid: lc_regrids[lc][tile] -> np.ndarray
lc_regrids = {lc: {} for lc in LC_MASKS}

#%%
tile = all_tiles[0]
for tile in all_tiles:

    print(f"\nProcessing tile {tile}...")
    tile_dir = os.path.join(MCD19_basedir, tile)
    file_pattern = os.path.join(
        tile_dir,
        f'MCD19A1.{tile}.061_01d_16day_CV-MVC_QCfilt_{QC_option}_*_rect.nc'
    )
    tile_files = sorted(glob.glob(file_pattern))
    print(f"  Found {len(tile_files)} year-composite files for tile {tile}")

    if len(tile_files) == 0:
        continue

    tile_records = []

    f_path = tile_files[0]
    for f_path in tile_files:

        year, period = parse_year_period(f_path)
        if year is None:
            print(f"    Could not parse year/composite_period from {f_path}, skipping.")
            continue

        ds = xr.open_dataset(f_path, decode_coords='all').rename({'lat': 'y', 'lon': 'x'})

        lat_1d = ds['y'].values  # renamed dim, values are still true latitudes
        lon_1d = ds['x'].values  # renamed dim, values are still true longitudes
        ndvi_2d = ds['NDVI'].values

        lc_name = list(LC_MASKS.keys())[0]
        for lc_name in LC_MASKS:

            # Regrid this land-cover mask to this tile's grid, once per (lc, tile), cached.
            if tile not in lc_regrids[lc_name]:
                print(f"    Regridding {lc_name} mask to tile {tile} grid...")
                regridded = (
                    lc_fraction_native[lc_name]
                    .rio.write_crs("EPSG:4326")
                    .rio.reproject_match(ds, resampling=Resampling.bilinear, nodata=np.nan)
                )
                lc_regrids[lc_name][tile] = regridded.values

            mask_2d = lc_regrids[lc_name][tile]

            pix_df = extract_valid_pixels(ndvi_2d, mask_2d, lat_1d, lon_1d, LC_THRESHOLD)
            if pix_df is None:
                continue

            pix_df['year'] = year
            pix_df['composite_period'] = period
            pix_df['tile'] = tile
            pix_df['land_cover'] = lc_name
            pix_df = pix_df.rename(columns={'ndvi': 'mcd19_ndvi'})

            tile_records.append(pix_df)

        ds.close()
        del ds, lat_1d, lon_1d, ndvi_2d

    if len(tile_records) == 0:
        print(f"  No valid land-cover-masked pixels found anywhere in tile {tile}.")
        del tile_records
        gc.collect()
        continue

    tile_df = pd.concat(tile_records, ignore_index=True)
    tile_df['year'] = tile_df['year'].astype('int32')
    tile_df['composite_period'] = tile_df['composite_period'].astype('int32')
    print(f"  Tile {tile}: {len(tile_df)} total (pixel x period x land_cover) records. Writing to parquet dataset...")

    table = pa.Table.from_pandas(tile_df, preserve_index=False)
    pads.write_dataset(
        table,
        base_dir=MCD19_PARQUET_DIR,
        format='parquet',
        partitioning=MCD19_PARTITIONING,
        existing_data_behavior='overwrite_or_ignore',
        basename_template=f'{tile}-part-{{i}}.parquet',
    )

    del tile_records, tile_df, table
    gc.collect()

elapsed = time.time() - start_time
print(f"\nDone. Elapsed time: {elapsed/60:.1f} minutes.")
print(f"MCD19 land-cover-filtered NDVI dataset written to: {MCD19_PARQUET_DIR}")

# %%