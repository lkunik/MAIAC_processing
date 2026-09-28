#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Lewis Kunik - University of Utah
# Contact: lewis.kunik@utah.edu
#
# Miscellaneous checker: count QA_flag values in CV-MVC pre-fill composites to see whether any
# snow-flagged composites (QA_flag == 2) survive step 01 (_02_CV-MVC_pipeline/01_process_MCD19A1_CV-MVC_pre-fill_composites_batch.py).
#   QA_flag: 0 = no data, 1 = highest quality, 2 = passed QC but snow/ice detected (MAIAC Status_QA bits 3-4)
# Only winter/shoulder composite periods are checked (see composite_periods below).
#
# Usage: python misc_check_QA_flag_snow_values.py [QC_option] [tile1 tile2 ...]
#   defaults: QC_option = CloudFree_LowAOD_AllAdj, tiles = h11v02 h11v03
#

#%%
import os
import sys
import glob
import re
import numpy as np
import pandas as pd
import xarray as xr

QC_option = sys.argv[1] if len(sys.argv) > 1 else 'CloudFree_LowAOD_AllAdj'
MODIS_tiles = sys.argv[2:] if len(sys.argv) > 2 else ['h11v02', 'h11v03']

prefill_basedir = '/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/pre-fill/'
composite_periods = list(range(0, 6)) + list(range(19, 23))  # 0-5 and 19-22

# e.g. MCD19A1.h11v02.061_1km_16day_CV-MVC_QCfilt_CloudFree_LowAOD_AllAdj_2019-02.nc
file_re = re.compile(r'_(\d{4})-(\d{2})\.nc$')

rows = []
for MODIS_tile in MODIS_tiles:
    dat_dir = os.path.join(prefill_basedir, f'QCfilt_{QC_option}', MODIS_tile)
    files = sorted(glob.glob(os.path.join(dat_dir, f'MCD19A1.{MODIS_tile}.*_QCfilt_{QC_option}_*.nc')))
    print(f'{MODIS_tile}: {len(files)} files in {dat_dir}')

    for f in files:
        m = file_re.search(f)
        if m is None:
            continue
        year, period = int(m.group(1)), int(m.group(2))
        if period not in composite_periods:
            continue

        with xr.open_dataset(f) as ds:
            if 'QA_flag' not in ds:
                print(f'    no QA_flag variable: {os.path.basename(f)}')
                continue
            qa = ds['QA_flag'].values
        valid = np.isfinite(qa)
        vals, counts = np.unique(qa[valid], return_counts=True)
        count_dict = {int(v): int(c) for v, c in zip(vals, counts)}
        rows.append({'tile': MODIS_tile, 'year': year, 'period': period,
                     'n_flag0': count_dict.get(0, 0), 'n_flag1': count_dict.get(1, 0), 'n_flag2': count_dict.get(2, 0),
                     'other_values': {k: v for k, v in count_dict.items() if k not in (0, 1, 2)} or ''})
        if count_dict.get(2, 0) > 0:
            print(f'    QA_flag == 2 found: {os.path.basename(f)} ({count_dict[2]} pixels)')

df = pd.DataFrame(rows)
if df.empty:
    print('No matching files found.')
    sys.exit(0)

pd.set_option('display.width', 200)
pd.set_option('display.max_rows', 500)
print('')
print(df.to_string(index=False))
print('')
print('Totals by tile:')
print(df.groupby('tile')[['n_flag0', 'n_flag1', 'n_flag2']].sum().to_string())
print('')
n_files_with_2 = int((df['n_flag2'] > 0).sum())
print(f'{n_files_with_2}/{len(df)} files checked contain QA_flag == 2 (QC option {QC_option}, periods {composite_periods})')
