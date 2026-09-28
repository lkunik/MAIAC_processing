#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Lewis Kunik - University of Utah
# Contact: lewis.kunik@utah.edu
#
# Checks whether the point-level files saved by the 01*_save_MCD19_*.py
# scripts are up to date with the gridded MCD19 files they are extracted
# from. For each site and product, compares the newest last-modified time
# among the gridded input files against the point file's last-modified time.
#
# Usage: python 00_check_point_data_uptodate.py [QC_descr]
#        (QC_descr defaults to the value used in 01-submit_py.sh)

#%%
import os
import sys
import glob
from datetime import datetime as dt

import numpy as np

QC_descr = 'CloudFree_LowAOD_ClearAdj'

grid_basedir = '/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv'
point_basedir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/data/QC_PhotoSpec_sites_backup/CV-MVC'

# Same site definitions as in the 01*_save_MCD19_*.py scripts
site_info = {
    'US-NR1': {'start': '2017-01-01', 'end': '2018-12-31', 'tile': 'h09v04'},
    'OSBS':   {'start': '2021-01-01', 'end': '2022-12-31', 'tile': 'h10v06'},
    'Ca-Obs': {'start': '2018-01-01', 'end': '2022-12-31', 'tile': 'h11v03'},
    'DEJU':   {'start': '2019-01-01', 'end': '2021-12-31', 'tile': 'h11v02'},
    'US-UMB': {'start': '2018-01-01', 'end': '2019-12-31', 'tile': 'h12v04'},
    'US-Ne3': {'start': '2018-01-01', 'end': '2019-12-31', 'tile': 'h10v04'},
}

# script: (gridded subdir, point subdir, point file suffix, is_hist_avg)
products = {
    '01a (pre-fill)':               ('pre-fill',                 'pre-fill',                '',                        False),
    '01c (pre-fill_OutlierFilter)': ('pre-fill_OutlierFilter',   'pre-fill_OutlierFilter',  '',                        False),
    '01d (lm-fill)':                ('lm-fill',                  'lm-fill',                 '',                        False),
    '01e (hist_avg)':               ('pre-fill',                 'hist_avg',                '_hist_avg',               True),
    '01f (basic-fill)':             ('basic-fill',               'basic-fill',              '',                        False),
    '01g (hist_avg_OutlierFilter)': ('pre-fill_OutlierFilter',   'hist_avg_OutlierFilter',  '_hist_avg_OutlierFilter', True),
}


def list_gridded_files(grid_subdir, tile, years, is_hist_avg):
    grid_dir = os.path.join(grid_basedir, grid_subdir, f'QCfilt_{QC_descr}', tile)
    if is_hist_avg:
        return glob.glob(os.path.join(grid_dir, 'hist_avg', f'MCD19A1.{tile}.*_hist_avg_??.nc'))
    files = []
    for year in years:
        files += glob.glob(os.path.join(grid_dir, f'MCD19A1.{tile}.*_{year}-??.nc'))
    return files


def fmt(ts):
    return dt.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M') if ts is not None else '-'


# def main():
print(f'QC_descr = {QC_descr}\n')
header = f"{'Script':<30} {'Site':<8} {'N grid':>6}  {'Newest gridded':<16}  {'Point file':<16}  Status"
print(header)
print('-' * len(header))

stale = []
for product, (grid_subdir, point_subdir, suffix, is_hist_avg) in products.items():
    for site_name, info in site_info.items():
        years = np.arange(int(info['start'][:4]), int(info['end'][:4]) + 1)
        grid_files = list_gridded_files(grid_subdir, info['tile'], years, is_hist_avg)
        grid_mtime = max(os.path.getmtime(f) for f in grid_files) if grid_files else None

        point_file = os.path.join(point_basedir, point_subdir, f'QCfilt_{QC_descr}',
                                    f'MCD19_QCfilt_{QC_descr}{suffix}_{site_name}.nc')
        point_mtime = os.path.getmtime(point_file) if os.path.exists(point_file) else None

        if grid_mtime is None:
            status = 'NO GRIDDED FILES'
        elif point_mtime is None:
            status = '** MISSING POINT FILE **'
            stale.append((product, site_name))
        elif point_mtime < grid_mtime:
            status = '** OUT OF DATE **'
            stale.append((product, site_name))
        else:
            status = 'ok'

        print(f'{product:<30} {site_name:<8} {len(grid_files):>6}  {fmt(grid_mtime):<16}  {fmt(point_mtime):<16}  {status}')
    print()

if stale:
    print('Needs re-running:')
    for product, site_name in stale:
        print(f'  {product}  {site_name}')
else:
    print('All point files are up to date.')


# if __name__ == '__main__':
#     main()
