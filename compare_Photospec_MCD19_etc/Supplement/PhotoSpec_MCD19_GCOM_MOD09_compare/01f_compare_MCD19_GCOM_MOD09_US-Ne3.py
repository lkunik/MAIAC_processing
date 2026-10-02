#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Lewis Kunik - University of Utah
# Contact: lewis.kunik@utah.edu
#
# Supplement: tower (FloX) and MCD19 vs. each comparison
# satellite product individually at US-Ne3. One panel per product and index:
#   NDVI: MOD09, GCOM-C, MOD13, MCD43
#   CCI:  MOD09, GCOM-C
# Comparison products are drawn as lines. Input files, tower overpass-window
# filtering, daily averaging, and plot styling follow the 01_ scripts in
# 03_final_compare/ (no MCD19 sigma shading); marker sizes follow the older
# versions of these scripts (see old/).
# Saves <panel_dir>/<site_name>_<NDVI|CCI>_<product>_panel.png for assembly by
# 02_plot_supplement_product_panels.py.

#%%
import os
import numpy as np
import xarray as xr
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter
from matplotlib.ticker import FuncFormatter as FFmt

#########################################
# Site settings
#########################################

site_name = 'US-Ne3'
plot_title = 'US-Ne3'
panel_letters = ('i', 'j')          # (NDVI, CCI) for MOD09/GCOM-C; (MOD13, MCD43) for the NDVI-only products
figsize = (7, 3)
month_ticks = [1, 4, 7, 10]

winter_periods = [
    # Ground snow cover periods
    (pd.Timestamp('2017-12-24'), pd.Timestamp('2018-02-25')),
    (pd.Timestamp('2018-11-09'), pd.Timestamp('2019-03-15')),
    (pd.Timestamp('2019-12-15'), pd.Timestamp('2020-02-08')),
]

#########################################
# Filepaths
#########################################

QC_descr = "CloudFree_LowAOD_ClearAdj"

dat_pt_basedir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/data/QC_PhotoSpec_sites_backup/'
dat_pt_basedir_QC = os.path.join(dat_pt_basedir, f'CV-MVC/lm-fill/QCfilt_{QC_descr}')
MCD19_QC_file = os.path.join(dat_pt_basedir_QC, f'MCD19_QCfilt_{QC_descr}_{site_name}.nc')

product_files = {
    prod: os.path.join(dat_pt_basedir, prod, f'{prod}_pt_{site_name}.nc')
    for prod in ['MOD09', 'GCOM', 'MOD13', 'MCD43']
}

FloX_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/data/FloX/US-Ne3/'
FloX_2018_file = os.path.join(FloX_dir, 'DFlox_SIF_VIs_2018.csv')
FloX_2019_file = os.path.join(FloX_dir, 'DFlox_SIF_VIs_2019.csv')

# MCD19 mean overpass time at US-Ne3 is 12:04 local mean solar time (LMST). FloX DateTime is UTC, so
# convert to LMST (UTC + lon/15) before selecting FloX times within +/- 1 hr of overpass
FloX_utc_to_LMST = pd.Timedelta(hours=-96.4397 / 15)
overpass_window = ('11:04', '13:04')
overpass_window_times = tuple(pd.Timestamp(t).time() for t in overpass_window)

plot_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/plots/CVMVC_QC_regression_stats/final/supplement/'
panel_dir = os.path.join(plot_dir, 'panels')
os.makedirs(panel_dir, exist_ok=True)

#########################################
# Plot constants
#########################################

MCD19_QC_color = '#BF2665'

product_styles = {
    # product: (legend label, color, marker, markersize)
    'MOD09': ('MOD09', '#B288E9', '^', np.sqrt(60)),
    'GCOM': ('GCOM-C', '#56B4E9', '^', 9),
    'MOD13': ('MOD13', '#0DA10A', 's', np.sqrt(35)),
    'MCD43': ('MCD43', '#B5C230', 's', np.sqrt(22)),
}

# (product, element, index into panel_letters)
panels = [
    ('MOD09', 'NDVI', 0),
    ('MOD09', 'CCI', 1),
    ('GCOM', 'NDVI', 0),
    ('GCOM', 'CCI', 1),
    ('MOD13', 'NDVI', 0),
    ('MCD43', 'NDVI', 1),
]

#########################################
# Functions
#########################################

def remove_running_window_outliers(df, window='14D', value_columns=('CCI', 'NDVI')):
    """Remove rows containing values outside running 1st/99th percentiles."""
    df = df.sort_values('DateTime').copy()
    indexed = df.set_index('DateTime')
    keep = pd.Series(True, index=indexed.index)
    for column in value_columns:
        if column not in indexed:
            continue
        values = indexed[column]
        lower = values.rolling(window, center=True, min_periods=2).quantile(0.01)
        upper = values.rolling(window, center=True, min_periods=2).quantile(0.99)
        keep &= values.isna() | ((values >= lower) & (values <= upper))
    return indexed.loc[keep.to_numpy()].reset_index()


def load_flox_year(path, year, doy_offset=0):
    df = pd.read_csv(path)
    if doy_offset:
        df['DoY'] = df['DoY'] - doy_offset
    df = df[df['DoY'].between(1, 366)]
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df = df[df['DateTime'].between(f'{year}-01-01', f'{year}-12-31')]
    df = df[(df['DateTime'] + FloX_utc_to_LMST).dt.time.between(*overpass_window_times)]
    df = df[df['NDVI'].between(0, 1)]
    return remove_running_window_outliers(df)


def load_reference():
    """
    Return (window times, filtered times, filtered daily DataFrame) of the tower record.
    FloX obs are restricted to the overpass window, then averaged to daily means.
    """
    df = pd.concat([
        load_flox_year(FloX_2018_file, 2018),
        load_flox_year(FloX_2019_file, 2019, doy_offset=365),
    ], ignore_index=True)
    df['Time'] = pd.to_datetime(df['DateTime']).dt.normalize()
    df = df.groupby('Time', as_index=False).mean(numeric_only=True)
    times = pd.to_datetime(df['Time'])
    return times, times, df


def to_datetimeindex(times):
    """
    Convert datetime64 or cftime times to a pandas DatetimeIndex. cftime dates (e.g. the
    Julian-calendar MOD13 times at some sites) keep their year/month/day components.
    """
    times = np.asarray(times)
    if times.size and not np.issubdtype(times.dtype, np.datetime64):
        return pd.to_datetime([t.isoformat() for t in times])
    return pd.to_datetime(times)


def break_at_gaps(times, vals):
    """
    Insert NaNs where the time step is well above the product's usual spacing, so
    lines are not drawn across data gaps (e.g. MOD13 records missing over winter).
    """
    times = to_datetimeindex(times)
    vals = np.asarray(vals, dtype=float).squeeze()
    if len(times) < 3:
        return times, vals
    order = np.argsort(times)
    times, vals = times[order], vals[order]
    steps = np.diff(times.values)
    gap = steps > 1.5 * np.median(steps)
    insert_at = np.where(gap)[0] + 1
    times = np.insert(times.values, insert_at, times.values[insert_at - 1] + steps[gap] / 2)
    vals = np.insert(vals, insert_at, np.nan)
    return pd.to_datetime(times), vals


def plot_product_panel(element, prod, ref_times, ref_df, mcd19_times, mcd19,
                       prod_times, prod_vals, title, xlim, outfile):
    label, color, marker, markersize = product_styles[prod]

    fig, ax = plt.subplots(figsize=figsize)

    for start, end in winter_periods:
        ax.axvspan(start, end, color='lightblue', alpha=0.5, zorder=0)
    ax.scatter(ref_times, ref_df[element], color='black', s=12, alpha=0.8)

    prod_times, prod_vals = break_at_gaps(prod_times, prod_vals)
    ax.plot(prod_times, prod_vals, color=color, marker=marker, markersize=markersize, linewidth=2.5,
            alpha=1, markeredgecolor='black', markeredgewidth=0.5, zorder=2)

    ax.plot(mcd19_times, mcd19, color=MCD19_QC_color, marker='o', markersize=8, linewidth=3.5,
            alpha=1, markeredgecolor='black', markeredgewidth=0.5, zorder=3)

    # y-limits from MCD19 and the tower record (as in the 03_final_compare panels)
    y_arrays = [mcd19, ref_df[element].values]
    y_min = np.nanmin(np.concatenate(y_arrays))
    y_max = np.nanmax(np.concatenate(y_arrays))
    y_range = y_max - y_min
    pad_lo = 0.15 if element == 'NDVI' else 0.05
    ax.set_ylim(y_min - pad_lo * y_range, y_max + 0.05 * y_range)
    ax.yaxis.set_major_formatter(FFmt(lambda y, _: f'{y:.1f}'))

    ax.set_xlim(*xlim)
    ax.xaxis.set_major_locator(MonthLocator(bymonth=month_ticks))
    ax.xaxis.set_major_formatter(DateFormatter('%b\n%Y'))
    ax.set_title(title, loc='center')
    ax.set_ylabel(element)

    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close(fig)

#########################################
# Main
#########################################
#%%
def main():

    MCD19_QC_pt = xr.open_dataset(MCD19_QC_file)
    product_pts = {prod: xr.open_dataset(f) for prod, f in product_files.items()}

    ref_times, ref_times_filtered, ref_df_filtered = load_reference()

    # Select MCD19 data within the time range of the tower data
    MCD19_QC_pt = MCD19_QC_pt.sel(time=slice(
        ref_times.iloc[0] - pd.Timedelta(days=16),
        ref_times.iloc[-1] + pd.Timedelta(days=16),
    ))
    MCD19_QC_times = np.array([np.datetime64(t) for t in MCD19_QC_pt['time'].values])

    xlim = (ref_times_filtered.min() - pd.Timedelta(days=10),
            ref_times_filtered.max() + pd.Timedelta(days=10))

    plt.rcParams.update({
        'axes.labelsize': 18,
        'axes.titlesize': 21,
        'xtick.labelsize': 17,
        'ytick.labelsize': 17,
        'legend.fontsize': 13
    })

    for prod, element, letter_idx in panels:
        prod_pt = product_pts[prod]
        if element not in prod_pt:
            print(f'{site_name} {prod}: no {element}, skipping')
            continue
        outfile = os.path.join(panel_dir, f'{site_name}_{element}_{prod}_panel.png')
        plot_product_panel(
            element, prod, ref_times_filtered, ref_df_filtered,
            MCD19_QC_times, MCD19_QC_pt[element].values,
            prod_pt['time'].values, prod_pt[element].values,
            f'{panel_letters[letter_idx]})  {plot_title}', xlim, outfile,
        )
        print(f'Saved {outfile}')


if __name__ == "__main__":
    main()
