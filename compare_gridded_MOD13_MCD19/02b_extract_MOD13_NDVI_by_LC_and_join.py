#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Lewis Kunik - University of Utah
# Contact: lewis.kunik@utah.edu
#
# Step 2 of 2 for the MCD19-vs-MOD13 gridded NDVI cross-comparison (ESSD manuscript, Sec 3.2).
#
# Purpose: for every MOD13 region ('WUS','EUS','CAm','Boreal') / year / 16-day composite
# period, extract valid NDVI pixels, and look up matching (lat, lon, year, composite_period)
# entries in the land-cover-filtered MCD19 dataset produced by
# 07a_extract_MCD19_ndvi_by_landcover.py. Matches (present in both products, same land-cover
# class membership already encoded from the MCD19 side) are written out to a final paired
# Parquet dataset, partitioned by land_cover, ready for the follow-up 2D histogram /
# linear-regression script.
#
# See the header of 07a_extract_MCD19_ndvi_by_landcover.py for the full set of methodological
# decisions this pipeline is built on (no ecoregion stratification, land-cover-only, exact
# coordinate matching at 3 decimal places, no NDVI scale factor).
#
# COMPOSITE-PERIOD DERIVATION: MOD13's time coordinate is decoded (units "days since
# 2000-01-01", calendar "julian") and its day-of-year is used to compute composite_period
# directly as floor((doy-1)/16), rather than trusting the file's raw time-dimension index.
# This matters because MODIS wasn't fully operational until roughly Feb/Mar 2000, so that
# year's MOD13 files don't start at time index 0 == composite_period 0 the way every later
# year's files do. A warning is printed if a computed composite_period falls outside 0-22,
# or if the same composite_period is computed more than once within one file (either would
# indicate a real misalignment worth checking).
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
import pandas as pd

import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.compute as pc

import time

#%%
#########################################
# Config
#########################################

MOD13_basedir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/MODIS/NDVI_EVI_MOD13A2/01d_rect/'
MOD13_REGIONS = ['WUS', 'EUS', 'CAm', 'Boreal']

ROUND_DECIMALS = 3  # must match 07a_extract_MCD19_ndvi_by_landcover.py

OUTPUT_ROOT = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/data/MCD19_MOD13_comparison'
MCD19_PARQUET_DIR = os.path.join(OUTPUT_ROOT, 'mcd19_ndvi_by_landcover')
PAIRED_PARQUET_DIR = os.path.join(OUTPUT_ROOT, 'mcd19_mod13_ndvi_paired')
Path(PAIRED_PARQUET_DIR).mkdir(parents=True, exist_ok=True)

# MUST match the partitioning used when this dataset was written in
# 07a_extract_MCD19_ndvi_by_landcover.py (explicit schema, not inferred, for robustness
# across pyarrow versions).
MCD19_PARTITIONING = pads.partitioning(
    pa.schema([("year", pa.int32()), ("composite_period", pa.int32())]),
    flavor="hive",
)
PAIRED_PARTITIONING = pads.partitioning(
    pa.schema([("land_cover", pa.string())]),
    flavor="hive",
)

#########################################
# Helper functions
#########################################

MOD13_FILENAME_RE_TEMPLATE = r'MOD13A2\.061_01d_aid0001_{region}_(?P<year>\d{{4}})\.nc$'


def get_mod13_files(base_dir, region):
    """Find all MOD13 files for a given region, keyed by year."""
    pattern = os.path.join(base_dir, f'MOD13A2.061_01d_aid0001_{region}_*.nc')
    files = sorted(glob.glob(pattern))
    year_re = re.compile(MOD13_FILENAME_RE_TEMPLATE.format(region=re.escape(region)))
    out = {}
    for f in files:
        m = year_re.search(os.path.basename(f))
        if m:
            out[int(m.group('year'))] = f
    return out


def get_dayofyear(t):
    """Return day-of-year for either a cftime object or a numpy/pandas datetime64 scalar."""
    if hasattr(t, 'dayofyr'):       # cftime datetime (non-standard calendars, e.g. 'julian')
        return int(t.dayofyr)
    ts = pd.Timestamp(t)
    return int(ts.dayofyear)


def extract_valid_pixels(ndvi_2d, lat_1d, lon_1d):
    """
    Vectorized extraction of (lat, lon, ndvi) for all non-NaN pixels in a 2D NDVI slice.
    Returns a DataFrame with columns ['lat', 'lon', 'mod13_ndvi'], or None if all-NaN.
    """
    valid = ~np.isnan(ndvi_2d)
    if not np.any(valid):
        return None
    rows, cols = np.where(valid)
    df = pd.DataFrame({
        'lat': np.round(lat_1d[rows], ROUND_DECIMALS),
        'lon': np.round(lon_1d[cols], ROUND_DECIMALS),
        'mod13_ndvi': ndvi_2d[rows, cols],
    })
    return df


def read_mcd19_year(year):
    """Read the MCD19 land-cover-filtered dataset for a single year (all tiles/periods/classes)."""
    dataset = pads.dataset(MCD19_PARQUET_DIR, format='parquet', partitioning=MCD19_PARTITIONING)
    filt = pc.field('year') == pa.scalar(year, type=pa.int32())
    table = dataset.to_table(filter=filt)
    df = table.to_pandas()
    return df


#########################################
# Begin Main
#########################################
#%%

start_time = time.time()

# Discover which years are actually available, per region.
mod13_files_by_region = {region: get_mod13_files(MOD13_basedir, region) for region in MOD13_REGIONS}
all_years = sorted(set().union(*[set(d.keys()) for d in mod13_files_by_region.values()]))
print(f"Found MOD13 files spanning years: {all_years[0]}-{all_years[-1]} ({len(all_years)} years total)")

#%%
year = all_years[0]
for year in all_years:

    print(f"\nProcessing year {year}...")

    mcd19_year_df = read_mcd19_year(year)
    if len(mcd19_year_df) == 0:
        print(f"  No MCD19 land-cover-filtered records found for year {year}, skipping.")
        continue
    print(f"  Loaded {len(mcd19_year_df)} MCD19 records for year {year} (all tiles/periods/land-cover classes).")

    year_matches = []

    region = MOD13_REGIONS[0]
    for region in MOD13_REGIONS:

        mod13_file = mod13_files_by_region[region].get(year)
        if mod13_file is None:
            print(f"    No MOD13 file for region {region}, year {year}, skipping.")
            continue

        print(f"    Processing region {region}...")
        ds = xr.open_dataset(mod13_file, decode_coords='all')

        lat_1d = ds['y'].values
        lon_1d = ds['x'].values
        n_time = ds.sizes['time']

        time_index = ds.indexes['time']
        seen_periods = set()

        t = 0
        for t in range(n_time):

            # Derive composite_period from the actual day-of-year rather than trusting the
            # raw time-dimension index. Necessary because MODIS wasn't fully online until
            # roughly Feb/Mar 2000, so year-2000 MOD13 files don't start at time index 0 ==
            # composite_period 0 the way every later year's files do - the index and the
            # true composite_period diverge for that year.
            doy = get_dayofyear(time_index[t])
            composite_period = (doy - 1) // 16

            if not (0 <= composite_period <= 22):
                print(
                    f"    WARNING: computed composite_period {composite_period} outside "
                    f"expected range 0-22 (time index {t}, day-of-year {doy}, file "
                    f"{os.path.basename(mod13_file)}). Skipping this time step."
                )
                continue

            if composite_period in seen_periods:
                print(
                    f"    WARNING: composite_period {composite_period} computed more than "
                    f"once in {os.path.basename(mod13_file)} (time index {t}, day-of-year "
                    f"{doy}). Check for duplicate/misaligned time steps in this file."
                )
            seen_periods.add(composite_period)

            ndvi_2d = ds['NDVI'].isel(time=t).values

            mod13_df = extract_valid_pixels(ndvi_2d, lat_1d, lon_1d)
            if mod13_df is None:
                continue

            mcd19_subset = mcd19_year_df[mcd19_year_df['composite_period'] == composite_period]
            if len(mcd19_subset) == 0:
                continue

            matched = pd.merge(mcd19_subset, mod13_df, on=['lat', 'lon'], how='inner')
            if len(matched) == 0:
                continue

            matched['region'] = region
            year_matches.append(matched)

        ds.close()
        del ds, lat_1d, lon_1d, time_index
        gc.collect()

    if len(year_matches) == 0:
        print(f"  No MCD19/MOD13 matches found for year {year}.")
        del mcd19_year_df, year_matches
        gc.collect()
        continue

    year_df = pd.concat(year_matches, ignore_index=True)

    # A pixel near a region-boundary buffer could in principle appear in >1 MOD13 region file;
    # de-duplicate on the join key + land_cover, keeping the first match.
    n_before = len(year_df)
    year_df = year_df.drop_duplicates(subset=['lat', 'lon', 'year', 'composite_period', 'land_cover'], keep='first')
    n_dropped = n_before - len(year_df)
    if n_dropped > 0:
        print(f"  Dropped {n_dropped} duplicate cross-region matches out of {n_before}.")

    print(f"  Year {year}: {len(year_df)} paired MCD19/MOD13 records. Writing to parquet dataset...")

    table = pa.Table.from_pandas(year_df, preserve_index=False)
    pads.write_dataset(
        table,
        base_dir=PAIRED_PARQUET_DIR,
        format='parquet',
        partitioning=PAIRED_PARTITIONING,
        existing_data_behavior='overwrite_or_ignore',
        basename_template=f'{year}-part-{{i}}.parquet',
    )

    del mcd19_year_df, year_matches, year_df, table
    gc.collect()

elapsed = time.time() - start_time
print(f"\nDone. Elapsed time: {elapsed/60:.1f} minutes.")
print(f"Paired MCD19/MOD13 NDVI dataset written to: {PAIRED_PARQUET_DIR}")

# %%