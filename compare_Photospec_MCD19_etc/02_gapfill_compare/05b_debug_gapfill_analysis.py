#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Per-tile checks of the gap-fill QC chain (diagnostics files -> 10_MCD19A1_CV-MVC_Annualize_GapfillFlag_counts.py annualized files -> map)
# Usage: python check_gapfill_QC.py [sample_year]
#

#%%
import os
import sys
import glob
import numpy as np
import xarray as xr

QC_option = 'CloudFree_LowAOD_ClearAdj'
sample_year = 2017
var = 'CCI'
base_dir = '/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/'

MODIS_tiles = [# "h07v06" 
# "h08v03",
"h08v04",
"h08v05",
"h08v06",
# "h08v07",
# "h09v02",
# "h09v03" ,
"h09v04", # US-NR1
"h09v05", # ecoregions related to US-NR1
# "h09v06",
# "h09v07",
# "h09v08" ,
"h10v02",
"h10v03",
"h10v04", # ecoregions related to US-NR1
"h10v05",
"h10v06", # OSBS
# "h10v07"
# "h10v08"
"h11v02", # DEJU
"h11v03", # Ca-Obs
"h11v04",
"h11v05",
# "h11v06",
# "h11v07",
# "h11v08",
# "h12v01",
"h12v02",
"h12v03",
"h12v04",
"h12v05",
"h10v02",
"h10v03",
"h10v04", # ecoregions related to US-NR1
"h10v05",
"h10v06", # OSBS
# "h10v07",
# "h10v08",
"h11v02", # DEJU
"h11v03", # Ca-Obs
"h11v04",
"h11v05",
# "h11v06",
# "h11v07",
# "h11v08",
# "h12v01",
"h12v02",
"h12v03",
"h12v04",
"h12v05",
# "h13v01",
"h13v02",
"h13v03",
"h13v04",
# "h14v01",
"h14v02",
"h14v03",
# "h14v04",
# "h15v01",
# "h15v02",
# "h16v01",
]


def n_files(subdir, tile, pattern):
    return len(glob.glob(os.path.join(base_dir, f'{subdir}/QCfilt_{QC_option}/{tile}/', pattern)))


for tile in MODIS_tiles:
    print(f'\n===== {tile} =====')

    # 1. inputs to the diagnostics script for the sample year (should be 23 each, 20 in 2000)
    n_pre = n_files('pre-fill_OutlierFilter', tile, f'MCD19A1.{tile}.*{sample_year}-??.nc')
    n_lm = n_files('lm-fill', tile, f'MCD19A1.{tile}.*{sample_year}-??.nc')
    n_basic = n_files('basic-fill', tile, f'MCD19A1.{tile}.*{sample_year}-??.nc')
    n_hist = len(glob.glob(os.path.join(base_dir, f'pre-fill/QCfilt_{QC_option}/{tile}/hist_avg/MCD19A1.{tile}.*.nc')))
    print(f'{sample_year} input files: pre-fill_OutlierFilter={n_pre}, lm-fill={n_lm}, basic-fill={n_basic}, hist_avg={n_hist}')

    # 2. per-year diagnostics files
    diag_dir = os.path.join(base_dir, f'QC_gapfill/QCfilt_{QC_option}/{tile}/')
    diag_files = sorted(glob.glob(os.path.join(diag_dir, f'MCD19A1.{tile}.061_1km_16day_CV-MVC_gapfillQC_QCfilt_{QC_option}_*.nc')))
    diag_years = [int(os.path.basename(f).split('_')[-1].split('.')[0]) for f in diag_files]
    print(f'diagnostics files: {len(diag_files)} years' + (f' ({min(diag_years)}-{max(diag_years)})' if diag_years else ''))

    # 3. sample-year diagnostics: gap-fill flag per composite period (# filled in thousands, '-' = whole period missing)
    f_sample = os.path.join(diag_dir, f'MCD19A1.{tile}.061_1km_16day_CV-MVC_gapfillQC_QCfilt_{QC_option}_{sample_year}.nc')
    if os.path.exists(f_sample):
        with xr.open_dataset(f_sample) as ds:
            flag = ds[f'gapfill_{var}'].values
            has_latlon = ('lat' in ds.variables) and ('lon' in ds.variables)
        per_cp = []
        for cp in range(flag.shape[0]):
            if not np.isfinite(flag[cp]).any():
                per_cp.append('-')
            else:
                per_cp.append(f'{(flag[cp] == 1).sum() / 1000:.0f}')
        n1, n0 = int((flag == 1).sum()), int((flag == 0).sum())
        print(f'{sample_year} gapfill_{var}: filled={n1}, original={n0}, missing={int(np.isnan(flag).sum())}; lat/lon in file: {has_latlon}')
        print(f'  filled per period (x1000): {" ".join(per_cp)}')
    else:
        print(f'{sample_year} diagnostics file not found')

    # 4. annualized file (output of pipeline step 10)
    ann_file = os.path.join(diag_dir, 'annualized', f'MCD19A1.{tile}.061_1km_annual_CV-MVC_QCfilt_{QC_option}_gapfillQC.nc')
    if not os.path.exists(ann_file):
        print('annualized file: NOT FOUND')
        continue
    with xr.open_dataset(ann_file) as ads:
        cnt = ads[f'gapfill_{var}_count'].values
        lat = ads['lat'].values
        lon = ads['lon'].values
        attrs = dict(ads.attrs)
    print(f'annualized: years {attrs.get("years_averaged")}; lat/lon NaN fraction = {np.isnan(lat).mean():.3f}/{np.isnan(lon).mean():.3f}')
    print(f'  gapfill_{var}_count: min={np.nanmin(cnt):.2f}, median={np.nanmedian(cnt):.2f}, '
          f'99th pct={np.nanpercentile(cnt, 99):.2f}, fraction == 0: {(cnt == 0).mean():.3f}')
    print(f'  composites without gap-fill flags per year: {attrs.get("composites_without_gapfill_flags_per_year")}')

    # 5. lat/lon of the annualized file vs. the QC_OutlierFilter annual file (should match exactly)
    QC_files = sorted(glob.glob(os.path.join(base_dir, f'QC_OutlierFilter/QCfilt_{QC_option}/{tile}/*_annual_OutlierFilterQC_*.nc')))
    if QC_files:
        with xr.open_dataset(QC_files[0]) as qds:
            dlat = np.nanmax(np.abs(qds['lat'].values - lat)) if np.isfinite(lat).any() else np.nan
        print(f'  max |lat difference| vs. OutlierFilterQC file: {dlat}')