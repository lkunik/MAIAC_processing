#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Lewis Kunik - University of Utah
# Contact: lewis.kunik@utah.edu
#
# For one MODIS tile: summarize the per-year gap-filling diagnostics files (QC_gapfill) into one file with
# the mean annual number, and mean annual fraction, of 16-day composites that were gap-filled (lm-fill) at each pixel.
#   gapfill_{VI}_count : mean across years of (number of composites with gapfill_{VI} == 1)
#   gapfill_{VI}_frac  : mean across years of (number of composites with gapfill_{VI} == 1) / (number of composites that exist that year)
# Same structure as 05_MCD19A1_CV-MVC_Annualize_OutlierFilter_counts.py.
#
# Usage: python 10_MCD19A1_CV-MVC_Annualize_GapfillFlag_counts.py <MODIS_tile> <QC_option>
#

#########################################
# Import packages
#########################################
#%%
import os
import sys
import glob
from pathlib import Path
import gc

import numpy as np
import xarray as xr

import time
from datetime import datetime as dt


#########################################
# Define Global Variables and constants
#########################################

analysis_years = np.arange(2000, 2026)
vi_vars = ['NDVI', 'CCI']
gapfill_vars = [f'gapfill_{v}' for v in vi_vars]

base_dir = '/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/'


#########################################
# Begin main
#########################################
#%%

def main():

    start_total = time.time()

    MODIS_tile = sys.argv[1] #'h09v04'
    QC_option = sys.argv[2] # 'CloudFree_LowAOD_ClearAdj'

    # per-year gap-filling diagnostics files (from 09_MCD19A1_GapFill_QC_diagnostics_batch.py)
    dat_dir = os.path.join(base_dir, f'QC_gapfill/QCfilt_{QC_option}/{MODIS_tile}/')
    dat_files = sorted(glob.glob(f'{dat_dir}/MCD19A1.{MODIS_tile}.061_1km_16day_CV-MVC_gapfillQC_QCfilt_{QC_option}_*.nc'))
    # pre-fill_OutlierFilter composites, used to count how many composites exist per year (denominator) and for lat/lon if needed
    composite_dir = os.path.join(base_dir, f'pre-fill_OutlierFilter/QCfilt_{QC_option}/{MODIS_tile}/')

    print(f'Processing tile {MODIS_tile}...')

    out_dir = os.path.join(dat_dir, 'annualized')
    outfile = os.path.join(out_dir, f'MCD19A1.{MODIS_tile}.061_1km_annual_CV-MVC_QCfilt_{QC_option}_gapfillQC.nc')
    if os.path.exists(outfile):
        print(f'Output file already exists: {outfile}. Overwriting.')

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # compose a list of years from the filenames (one file per year)
    years = []
    for fname in dat_files:
        basename = os.path.basename(fname)
        # Example: 'MCD19A1.h09v04.061_1km_16day_CV-MVC_gapfillQC_QCfilt_CloudFree_LowAOD_ClearAdj_2017.nc'
        year = int(basename.split('_')[-1].split('.')[0])
        years.append(year)
    years = np.array(years)

    count_arr = []
    frac_arr = []
    n_composites_dict = {}   # number of composites that exist in each year (denominator)
    n_empty_dict = {}        # number of existing composites whose gap-fill flags are entirely missing, per year
    latlon = None

    for year in analysis_years:
        print(f'Processing year {year}...')
        if year not in years:
            print(f'No gap-fill QC file for {year}. Skipping this year.')
            continue
        f = dat_files[np.where(years == year)[0][0]]

        # denominator: number of pre-fill_OutlierFilter composite files that exist for this tile/year
        composite_files_year = sorted(glob.glob(os.path.join(
            composite_dir, f'MCD19A1.{MODIS_tile}.061_1km_16day_CV-MVC_QCfilt_{QC_option}_OutlierFilter_{year}-??.nc')))
        existing_cps = [int(os.path.basename(cf).split('-')[-1].split('.')[0]) for cf in composite_files_year]
        n_composites = len(existing_cps)
        print(f'{n_composites} composites exist for {year}')
        if n_composites == 0:
            print(f'No composite files for {year}. Skipping this year.')
            continue
        n_composites_dict[year] = n_composites

        with xr.open_dataset(f) as ds:
            if 'NDVI_rank' in ds:
                ds = ds.drop_vars('NDVI_rank')
            if 'VZA_rank' in ds:
                ds = ds.drop_vars('VZA_rank')
            dat_xr = ds[gapfill_vars].load()   # -1 fill value decodes to NaN (no value after lm-fill, or period not processed)

        # sanity check: existing composites whose gap-fill flags are entirely missing (e.g. lm-fill, basic-fill or hist_avg
        # file missing for that period in the diagnostics script); these count as 0 filled but are still in the denominator
        empty_cps = [cp for cp in existing_cps
                     if not np.isfinite(dat_xr[gapfill_vars[0]].sel(composite_period=cp).values).any()]
        n_empty_dict[year] = len(empty_cps)
        if len(empty_cps) > 0:
            print(f'WARNING: {year}: composite periods {empty_cps} exist in pre-fill_OutlierFilter but have no gap-fill flags')

        # number of composites gap-filled in this year (NaN flags count as not filled)
        n_filled = (dat_xr == 1).sum(dim='composite_period').astype(np.float32)
        count_arr.append(n_filled.expand_dims(year=[year]))
        frac_arr.append((n_filled / n_composites).astype(np.float32).expand_dims(year=[year]))

        # keep the 2D lat/lon (from the diagnostics file if present, otherwise from a composite file)
        if latlon is None:
            if ('lat' in dat_xr.coords) and ('lon' in dat_xr.coords):
                latlon = {'lat': dat_xr['lat'].load(), 'lon': dat_xr['lon'].load()}
            else:
                with xr.open_dataset(composite_files_year[0]) as cds:
                    latlon = {'lat': cds['lat'].load(), 'lon': cds['lon'].load()}
        del dat_xr
        gc.collect()

    if len(count_arr) == 0:
        print(f'No gap-fill QC data for tile {MODIS_tile}, exiting...')
        sys.exit(1)

    # average across years so the output has only spatial dimensions
    count_mean = xr.concat(count_arr, dim='year').mean(dim='year')
    frac_mean = xr.concat(frac_arr, dim='year').mean(dim='year')
    del count_arr, frac_arr
    gc.collect()

    out_ds = xr.Dataset()
    for v in vi_vars:
        out_ds[f'gapfill_{v}_count'] = count_mean[f'gapfill_{v}']
        out_ds[f'gapfill_{v}_frac'] = frac_mean[f'gapfill_{v}']
        out_ds[f'gapfill_{v}_count'].attrs = {
            'long_name': f'mean annual number of 16-day composites gap-filled (lm-fill) for {v}',
            'units': 'composites per year',
            'description': (f'For each year, number of composites with gapfill_{v} == 1 (missing in pre-fill_OutlierFilter, '
                            f'present in lm-fill); then averaged across years')}
        out_ds[f'gapfill_{v}_frac'].attrs = {
            'long_name': f'mean annual fraction of 16-day composites gap-filled (lm-fill) for {v}',
            'units': '1',
            'description': (f'For each year, number of composites with gapfill_{v} == 1 divided by the number of '
                            f'pre-fill_OutlierFilter composites that exist in that year; then averaged across years')}

    # drop any leftover non-spatial coordinates and attach the 2D lat/lon
    out_ds = out_ds.drop_vars([c for c in out_ds.coords if c not in ('y', 'x')])
    out_ds = out_ds.assign_coords(lat=latlon['lat'], lon=latlon['lon'])

    # define global attributes
    years_used = sorted(n_composites_dict.keys())
    out_ds.attrs['title'] = 'MODIS MCD19A1.061 MAIAC 16-day CV-MVC composites - gap-filling QC summary: mean annual number and fraction of composites gap-filled'
    out_ds.attrs['QC_option'] = QC_option
    out_ds.attrs['MODIS_tile'] = MODIS_tile
    out_ds.attrs['grid'] = 'MODIS 1 km sinusoidal tile grid (y, x); lat and lon are 2D pixel-center coordinates'
    out_ds.attrs['years_averaged'] = f'{years_used[0]}-{years_used[-1]}'
    out_ds.attrs['composites_per_year'] = ', '.join(f'{y}: {n_composites_dict[y]}' for y in years_used)
    out_ds.attrs['composites_without_gapfill_flags_per_year'] = ', '.join(f'{y}: {n_empty_dict[y]}' for y in years_used)
    out_ds.attrs['source_files'] = f'{dat_dir}MCD19A1.{MODIS_tile}.061_1km_16day_CV-MVC_gapfillQC_QCfilt_{QC_option}_YYYY.nc'
    out_ds.attrs['Citation'] = 'Lyapustin, A., Wang, Y. (2022). MODIS/Terra+Aqua Land Surface BRF Daily L2G Global 500m and 1km SIN Grid V061 [Data set]. NASA EOSDIS Land Processes Distributed Active Archive Center. Accessed from https://doi.org/10.5067/MODIS/MCD19A1.061'
    out_ds.attrs['Processed_by'] = 'Lewis Kunik, lewis.kunik@utah.edu'
    out_ds.attrs['Date_created'] = str(dt.today().date())

    # Finally, save to netcdf
    out_ds.to_netcdf(path=outfile, format="NETCDF4")
    out_ds.close()

    print(f'saved {outfile}')
    print(f'tile {MODIS_tile} done ({(time.time() - start_total)/60:.1f} min)')

if __name__ == "__main__":
    main()