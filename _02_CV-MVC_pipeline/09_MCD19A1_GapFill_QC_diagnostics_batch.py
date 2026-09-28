#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Lewis Kunik - University of Utah
# Contact: lewis.kunik@utah.edu
#
# Created on Wed Sep 23 2026
#
# Per-year gap-filling diagnostics for MCD19A1 CV-MVC composites.
# For one tile/year/QC option, loops over all 23 composite periods and writes one file with:
#   gapfill_NDVI, gapfill_CCI         : 1 = filled by lm-fill, 0 = original (pre-fill) value, missing = no value after lm-fill
#   NDVI_lmfill_minus_hist            : lm-fill NDVI minus historical mean NDVI for that composite period
#   CCI_lmfill_minus_hist             : lm-fill CCI minus historical mean CCI
#   NDVI_basicfill_minus_hist         : basic-fill NDVI minus historical mean NDVI
#   CCI_basicfill_minus_hist          : basic-fill CCI minus historical mean CCI
#
# Usage: python 09_MCD19A1_GapFill_QC_diagnostics_batch.py <MODIS_tile> <composite_year> <QC_option>
#

#########################################
# Import packages
#########################################
#%%
# File system packages
import os  # operating system library
import sys
import glob
from pathlib import Path

# runtime packages
import warnings
import gc

# numerical/data management packages
import numpy as np
import xarray as xr  # for multi dimensional data
import pandas as pd

# time and date packages
import time
from datetime import datetime as dt  # date time library
from datetime import timedelta


#########################################
# Define Global Filepaths
#########################################
#%%

base_dir = '/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/'


#########################################
# Define Global Variables and constants
#########################################

DOY_composite_starts = np.arange(1, 366, 16)
composite_periods = range(len(DOY_composite_starts))  # 0-22

vi_vars = ['NDVI', 'CCI']

# ---- Options that need confirmation (see notes) ----
# How to define the historical mean VI:
#   'from_mean_reflectance' : compute NDVI/CCI from the historical mean band 1, 2, 11 reflectance
#                             (consistent with how lm-fill and basic-fill construct filled values)
#   'file_variable'         : use NDVI/CCI variables stored in the hist_avg files (i.e., mean of VIs), if present
HIST_VI_METHOD = 'file_variable'

# How to get the pre-fill VI used for the gapfill flag:
#   'file_variable'    : use NDVI/CCI stored in the pre-fill_OutlierFilter files
#   'from_reflectance' : recompute NDVI/CCI from the pre-fill_OutlierFilter reflectance
PREFILL_VI_SOURCE = 'file_variable'


#########################################
# Define Global functions
#########################################

def open_squeezed(filename):
    ds = xr.open_dataset(filename)
    if 'time' in ds.dims:
        ds = ds.squeeze('time')
    return ds


def find_period_file(file_list, composite_year, composite_period):
    matches = [f for f in file_list if f.endswith(f'{composite_year}-{composite_period:02d}.nc')]
    if len(matches) == 0:
        return None
    if len(matches) > 1:
        print(f'WARNING: {len(matches)} files match {composite_year}-{composite_period:02d}, using {matches[0]}')
    return matches[0]


def find_hist_file(file_list, composite_period):
    matches = [f for f in file_list if f.endswith(f'{composite_period:02d}.nc')]
    if len(matches) == 0:
        return None
    return matches[0]


def compute_vis_from_reflectance(ds):
    ndvi = (ds['Sur_refl2'] - ds['Sur_refl1']) / (ds['Sur_refl2'] + ds['Sur_refl1'])
    cci = (ds['Sur_refl11'] - ds['Sur_refl1']) / (ds['Sur_refl11'] + ds['Sur_refl1'])
    return {'NDVI': ndvi, 'CCI': cci}


def get_vis(ds, source, label):
    if source == 'file_variable':
        missing_vars = [v for v in vi_vars if v not in ds.data_vars]
        if len(missing_vars) > 0:
            raise KeyError(f'{label} file is missing VI variable(s) {missing_vars}')
        vis = {v: ds[v] for v in vi_vars}
    elif source in ['from_reflectance', 'from_mean_reflectance']:
        vis = compute_vis_from_reflectance(ds)
    else:
        raise ValueError(f'Unknown VI source option: {source}')

    # drop scalar time coordinate so arithmetic across files (different timestamps) doesn't conflict
    return {v: da.drop_vars('time', errors='ignore') for v, da in vis.items()}


def composite_start_date(composite_year, composite_period):
    doy = int(DOY_composite_starts[composite_period])
    return np.datetime64(dt(composite_year, 1, 1) + timedelta(days=doy - 1), 'ns')


#########################################
# Begin main
#########################################
#%%
def main():

    try:
        # mark start time to keep track of elapsed
        start_total = time.time()

        MODIS_tile = sys.argv[1]  # 'h10v04'
        composite_year = int(sys.argv[2])  # command line argument
        QC_option = sys.argv[3]  # command line argument 

        # MODIS_tile = 'h09v04'
        # composite_year = 2017  # command line argument
        # QC_option = 'CloudFree_LowAOD_ClearAdj'


        prefill_dir = os.path.join(base_dir, f'pre-fill_OutlierFilter/QCfilt_{QC_option}/{MODIS_tile}/')
        lmfill_dir = os.path.join(base_dir, f'lm-fill/QCfilt_{QC_option}/{MODIS_tile}/')
        basicfill_dir = os.path.join(base_dir, f'basic-fill/QCfilt_{QC_option}/{MODIS_tile}/')
        hist_dir = os.path.join(base_dir, f'pre-fill/QCfilt_{QC_option}/{MODIS_tile}/hist_avg')

        prefill_files = sorted(glob.glob(os.path.join(prefill_dir, f'MCD19A1.{MODIS_tile}.*.nc')))
        lmfill_files = sorted(glob.glob(os.path.join(lmfill_dir, f'MCD19A1.{MODIS_tile}.*.nc')))
        basicfill_files = sorted(glob.glob(os.path.join(basicfill_dir, f'MCD19A1.{MODIS_tile}.*.nc')))
        hist_files = sorted(glob.glob(os.path.join(hist_dir, f'MCD19A1.{MODIS_tile}.*.nc')))

        out_dir = os.path.join(base_dir, f'QC_gapfill/QCfilt_{QC_option}/{MODIS_tile}')
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        outfile = os.path.join(out_dir, f'MCD19A1.{MODIS_tile}.061_1km_16day_CV-MVC_gapfillQC_QCfilt_{QC_option}_{composite_year}.nc')

        period_ds_list = []
        template_ds = None

        for composite_period in composite_periods:

            prefill_file = find_period_file(prefill_files, composite_year, composite_period)
            lmfill_file = find_period_file(lmfill_files, composite_year, composite_period)
            basicfill_file = find_period_file(basicfill_files, composite_year, composite_period)
            hist_file = find_hist_file(hist_files, composite_period)

            file_status = {'pre-fill_OutlierFilter': prefill_file, 'lm-fill': lmfill_file,
                            'basic-fill': basicfill_file, 'hist_avg': hist_file}
            missing_sources = [k for k, v in file_status.items() if v is None]
            if len(missing_sources) > 0:
                # expected for high-latitude winter periods skipped in lm-fill, and early 2000 periods
                print(f'Period {composite_period:02d}: missing {missing_sources}, writing missing values for this period')
                period_ds_list.append(None)
                continue

            print(f'Computing gap-fill diagnostics - composite year: {composite_year}, time period {composite_period}')

            prefill_ds = open_squeezed(prefill_file)
            lmfill_ds = open_squeezed(lmfill_file)
            basicfill_ds = open_squeezed(basicfill_file)
            hist_ds = open_squeezed(hist_file)

            prefill_vi = get_vis(prefill_ds, PREFILL_VI_SOURCE, 'pre-fill_OutlierFilter')
            lmfill_vi = get_vis(lmfill_ds, 'file_variable', 'lm-fill')
            basicfill_vi = get_vis(basicfill_ds, 'file_variable', 'basic-fill')
            hist_vi = get_vis(hist_ds, HIST_VI_METHOD, 'hist_avg')

            out_vars = {}
            for var in vi_vars:

                if prefill_vi[var].shape != lmfill_vi[var].shape:
                    raise ValueError(f'Grid mismatch for {var} between pre-fill {prefill_vi[var].shape} and lm-fill {lmfill_vi[var].shape}')

                prefill_missing = np.isnan(prefill_vi[var])
                lmfill_valid = ~np.isnan(lmfill_vi[var])

                # 1 where pre-fill was missing and lm-fill supplied a value, 0 where the original value was kept,
                # missing where lm-fill has no value (gap not filled, or no data)
                gapfill_flag = xr.where(prefill_missing & lmfill_valid, 1.0, 0.0).where(lmfill_valid)

                # sanity checks
                n_filled = int((gapfill_flag == 1).sum())
                n_original = int((gapfill_flag == 0).sum())
                n_lost = int(((~prefill_missing) & (~lmfill_valid)).sum())
                n_unfilled = int((prefill_missing & ~lmfill_valid & ~np.isnan(hist_vi[var])).sum())
                print(f'  {var}: {n_filled} filled, {n_original} original, {n_unfilled} still missing after lm-fill (hist mean available)')
                if n_lost > 0:
                    print(f'  WARNING: {var}: {n_lost} pixels valid in pre-fill but missing in lm-fill')

                out_vars[f'gapfill_{var}'] = gapfill_flag
                out_vars[f'{var}_lmfill_minus_hist'] = lmfill_vi[var] - hist_vi[var]
                out_vars[f'{var}_basicfill_minus_hist'] = basicfill_vi[var] - hist_vi[var]
                out_vars[f'{var}_lmfill_minus_basicfill'] = lmfill_vi[var] - basicfill_vi[var]

            period_ds = xr.Dataset(out_vars)
            if template_ds is None:
                template_ds = period_ds
            period_ds_list.append(period_ds)

            for ds in [prefill_ds, lmfill_ds, basicfill_ds, hist_ds]:
                ds.close()
            gc.collect()

        if template_ds is None:
            print(f'No complete sets of input files for tile {MODIS_tile}, year {composite_year}, exiting...')
            sys.exit(0)

        # fill periods without input files with missing values on the same grid
        period_ds_list = [xr.full_like(template_ds, np.nan) if ds is None else ds for ds in period_ds_list]

        out_ds = xr.concat(period_ds_list, dim=pd.Index(list(composite_periods), name='composite_period'))
        out_ds = out_ds.assign_coords(
            composite_start=('composite_period', [composite_start_date(composite_year, p) for p in composite_periods]))

        # variable attributes
        for var in vi_vars:
            out_ds[f'gapfill_{var}'].attrs['long_name'] = f'{var} gap-fill flag (lm-fill)'
            out_ds[f'gapfill_{var}'].attrs['flag_values'] = '0, 1'
            out_ds[f'gapfill_{var}'].attrs['flag_meanings'] = 'original_value gap_filled'
            out_ds[f'gapfill_{var}'].attrs['comment'] = 'Missing where no value exists after lm-fill'
            out_ds[f'{var}_lmfill_minus_hist'].attrs['long_name'] = f'lm-fill {var} minus historical mean {var} for composite period'
            out_ds[f'{var}_basicfill_minus_hist'].attrs['long_name'] = f'basic-fill {var} minus historical mean {var} for composite period'
            out_ds[f'{var}_lmfill_minus_hist'].attrs['hist_vi_method'] = HIST_VI_METHOD
            out_ds[f'{var}_basicfill_minus_hist'].attrs['hist_vi_method'] = HIST_VI_METHOD

        # define global attributes
        out_ds.attrs['title'] = 'MODIS MCD19A1.061 MAIAC CV-MVC 16-day composites - gap-filling diagnostics'
        out_ds.attrs['year'] = str(composite_year)
        out_ds.attrs['MODIS_tile'] = MODIS_tile
        out_ds.attrs['QC_option'] = QC_option
        out_ds.attrs['Comment1'] = 'gapfill flags derived by comparing pre-fill_OutlierFilter and lm-fill composites'
        out_ds.attrs['Comment2'] = f'historical mean VI method: {HIST_VI_METHOD}; pre-fill VI source: {PREFILL_VI_SOURCE}'
        out_ds.attrs['Citation'] = 'Lyapustin, A., Wang, Y. (2022). MODIS/Terra+Aqua Land Surface BRF Daily L2G Global 500m and 1km SIN Grid V061 [Data set]. NASA EOSDIS Land Processes Distributed Active Archive Center. Accessed from https://doi.org/10.5067/MODIS/MCD19A1.061'
        out_ds.attrs['Processed_by'] = 'Lewis Kunik, lewis.kunik@utah.edu'
        out_ds.attrs['Date_created'] = str(dt.today().date())

        # flags stored as int8 with -1 as fill, differences as float32
        encoding = {}
        for var in vi_vars:
            encoding[f'gapfill_{var}'] = {'dtype': 'int8', '_FillValue': -1, 'zlib': True}
            encoding[f'{var}_lmfill_minus_hist'] = {'dtype': 'float32', 'zlib': True}
            encoding[f'{var}_basicfill_minus_hist'] = {'dtype': 'float32', 'zlib': True}

        # Finally, save to netcdf
        out_ds.to_netcdf(path=outfile, format='NETCDF4', encoding=encoding)
        out_ds.close()

        print(f'Wrote gap-fill diagnostics to {outfile}')
        print(f'Total elapsed: {(time.time() - start_total)/60:.2f} min')


    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)        

if __name__ == '__main__':
   main()