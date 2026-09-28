#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Lewis Kunik - University of Utah
# Contact: lewis.kunik@utah.edu
#
# Created on Tue Oct 24 2023
#
# For one MODIS tile and one year: compute the difference of each 16-day pre-fill CV-MVC composite's
# CCI and NDVI from the historical (2000-2025) median for that composite period, and save it alongside
# the outlier filter masks (outlier_CCI, outlier_NDVI) so later steps can split values into flagged vs. kept.
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

# time and date packages
import time
from datetime import datetime as dt  # date time library


#########################################
# Begin main
#########################################
#%%

# mark start time to keep track of elapsed
start_total = time.time()

def main():

    try:
        # Command line arguments
        MODIS_tile = sys.argv[1] # 'h14v03'
        composite_year = int(sys.argv[2]) # 2017
        QC_option = sys.argv[3] #'CloudFree_LowAOD_ClearAdj'

        dat_dir = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/pre-fill/QCfilt_{QC_option}/{MODIS_tile}/'
        climatology_dir = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/pre-fill/QCfilt_{QC_option}/{MODIS_tile}/hist_avg/'
        QC_dir = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/QC_OutlierFilter/QCfilt_{QC_option}/{MODIS_tile}/'
        out_dir = os.path.join(QC_dir, 'diff_from_median')

        Path(out_dir).mkdir(parents=True, exist_ok=True)

        composite_periods = np.arange(0, 23)  # 0-22

        # names of the variables to read from each climatology file, per index:
        # key = index name used throughout this script, value = median variable name in the climatology file
        CLIM_VARS = {'NDVI': 'NDVI_median',
                     'CCI':  'CCI_median'}

        def dat_file_for(year, cp):
            # full path of the pre-fill CV-MVC composite file for a given year and composite period
            return os.path.join(dat_dir, f'MCD19A1.{MODIS_tile}.061_1km_16day_CV-MVC_QCfilt_{QC_option}_{year}-{cp:02d}.nc')

        def clim_file_for(cp):
            # full path of the climatology file for a given composite period
            return os.path.join(climatology_dir, f'MCD19A1.{MODIS_tile}.061_1km_CV-MVC_QCfilt_{QC_option}_hist_avg_{cp:02d}.nc')

        # annual outlier filter QC file for this tile/year (from 03_MCD19A1_CV-MVC_OutlierFilter.py)
        QC_file = os.path.join(QC_dir, f'MCD19A1.{MODIS_tile}.061_1km_CV-MVC_QCfilt_{QC_option}_annual_OutlierFilterQC_{composite_year}.nc')
        if not os.path.exists(QC_file):
            raise FileNotFoundError(f'Outlier filter QC file not found: {QC_file}')

        outfile = os.path.join(out_dir, f'MCD19A1.{MODIS_tile}.061_1km_CV-MVC_QCfilt_{QC_option}_diff_from_median_{composite_year}.nc')


        #########################################
        # Step 1: read the outlier masks and the tile grid from the QC file
        #########################################

        with xr.open_dataset(QC_file) as qds:
            ny, nx = qds.sizes['y'], qds.sizes['x']   # tile grid size (1200 x 1200)
            lat2d = qds['lat'].load()                 # 2D (y, x) pixel-center latitudes
            lon2d = qds['lon'].load()                 # 2D (y, x) pixel-center longitudes
            # (composite_period, y, x) int8 masks: 1 = composite removed because of that index, 0 = kept / missing / not testable
            outlier = {idx: qds[f'outlier_{idx}'].sel(composite_period=composite_periods).values.astype(np.int8)
                       for idx in CLIM_VARS}

        def read2d(ds, var):
            # reads one variable from an open dataset as a 2D numpy array of shape (ny, nx)
            arr = np.squeeze(ds[var].values)   # drop any length-1 dims (e.g. a single 'time' dim)
            assert arr.shape == (ny, nx), f'{var}: unexpected shape {arr.shape}'
            return arr


        #########################################
        # Step 2: difference from the historical median, one composite period at a time
        #########################################

        # (composite_period, y, x) float32 arrays, NaN wherever the composite value or the median is missing
        diff = {idx: np.full((len(composite_periods), ny, nx), np.nan, dtype=np.float32) for idx in CLIM_VARS}

        for i, cp in enumerate(composite_periods):
            f = dat_file_for(composite_year, cp)
            fc = clim_file_for(cp)
            if not os.path.exists(f):
                print(f'Missing data file (left NaN): {f}')
                continue
            if not os.path.exists(fc):
                print(f'Missing climatology file (left NaN): {fc}')
                continue

            with xr.open_dataset(f) as ds:
                # b1 = red (620-670 nm), b2 = NIR (841-876 nm), b11 = 526-536 nm
                b1, b2, b11 = (read2d(ds, v) for v in ('Sur_refl1', 'Sur_refl2', 'Sur_refl11'))
            with np.errstate(divide='ignore', invalid='ignore'):
                vals = {'NDVI': (b2 - b1) / (b2 + b1),
                        'CCI':  (b11 - b1) / (b11 + b1)}

            with xr.open_dataset(fc) as cds:
                for idx, med_var in CLIM_VARS.items():
                    diff[idx][i] = vals[idx] - read2d(cds, med_var)   # signed difference from the historical median

        # sanity check: every flagged outlier should have a valid difference (the flags were computed from these same values)
        for idx in CLIM_VARS:
            n_flag_nan = int(((outlier[idx] == 1) & ~np.isfinite(diff[idx])).sum())
            if n_flag_nan > 0:
                print(f'WARNING: {n_flag_nan} {idx} outlier flags where the {idx} difference from median is NaN')


        #########################################
        # Step 3: save
        #########################################

        dims = ('composite_period', 'y', 'x')
        out_ds = xr.Dataset(
            {'CCI_diff':     (dims, diff['CCI']),
             'NDVI_diff':    (dims, diff['NDVI']),
             'outlier_CCI':  (dims, outlier['CCI']),
             'outlier_NDVI': (dims, outlier['NDVI'])},
            coords={'composite_period': composite_periods,
                    'lat': lat2d,
                    'lon': lon2d})

        for idx in CLIM_VARS:
            out_ds[f'{idx}_diff'].attrs = {
                'long_name': f'{idx} minus historical median {idx} for this composite period',
                'units': '1',
                'description': (f'{idx} from the pre-fill CV-MVC composite minus {CLIM_VARS[idx]} from the '
                                f'hist_avg climatology file for the same composite period (signed); NaN where either is missing')}
            out_ds[f'outlier_{idx}'].attrs = {
                'long_name': f'outlier filter mask for {idx} (1 = composite removed because of {idx})',
                'units': '1',
                'flag_values': '0, 1',
                'flag_meanings': 'kept removed',
                'description': 'copied from the annual OutlierFilterQC file; 0 also where data were missing or not testable'}

        out_ds.attrs = {
            'title': 'MODIS MCD19A1.061 MAIAC 16-day CV-MVC composites - difference of CCI and NDVI from historical median, with outlier filter masks',
            'MODIS_tile': MODIS_tile,
            'year': int(composite_year),
            'QC_option': QC_option,
            'grid': 'MODIS 1 km sinusoidal tile grid (y, x); lat and lon are 2D pixel-center coordinates',
            'source_composites': f'{dat_dir}MCD19A1.{MODIS_tile}.061_1km_16day_CV-MVC_QCfilt_{QC_option}_{composite_year}-CP.nc',
            'source_climatology': f'{climatology_dir}MCD19A1.{MODIS_tile}.061_1km_CV-MVC_QCfilt_{QC_option}_hist_avg_CP.nc',
            'source_outlier_QC': QC_file,
            'Processed_by': 'Lewis Kunik, lewis.kunik@utah.edu',
            'Date_created': str(dt.today().date())}

        encoding = {v: {'zlib': True, 'complevel': 4} for v in out_ds.data_vars}
        out_ds.to_netcdf(path=outfile, format="NETCDF4", encoding=encoding)
        out_ds.close()
        print(f'saved {outfile}')
        print(f'elapsed: {time.time() - start_total:.1f} s')

    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
   main()