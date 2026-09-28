import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import numpy as np

MODIS_tile = 'h09v04'
data_dir = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/01d_rect/QCfilt_CloudFree/{MODIS_tile}/hist_avg/'
out_dir = f'/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/plots/maps/regrid_validation/QCfilt_CloudFree/hist_avg_checks/{MODIS_tile}'
os.makedirs(out_dir, exist_ok=True)

files = sorted([f for f in os.listdir(data_dir) if f.endswith('.nc')])

# Determine shared color ranges across all files
cci_min, cci_max = np.inf, -np.inf
ndvi_min, ndvi_max = np.inf, -np.inf

print("Computing color ranges...")
for fname in files:
    ds = xr.open_dataset(os.path.join(data_dir, fname))
    for var, vmin, vmax in [('CCI', cci_min, cci_max), ('NDVI', ndvi_min, ndvi_max)]:
        vals = ds[var].values
        mn = np.nanpercentile(vals, 2)
        mx = np.nanpercentile(vals, 98)
        if var == 'CCI':
            cci_min = min(cci_min, mn)
            cci_max = max(cci_max, mx)
        else:
            ndvi_min = min(ndvi_min, mn)
            ndvi_max = max(ndvi_max, mx)
    ds.close()

print(f"CCI range: {cci_min:.4f} - {cci_max:.4f}")
print(f"NDVI range: {ndvi_min:.4f} - {ndvi_max:.4f}")

for fname in files:
    doy_str = fname.split('_')[-1].replace('.nc', '')  # e.g. '07'
    fpath = os.path.join(data_dir, fname)
    ds = xr.open_dataset(fpath)
    
    for var, vmin, vmax, cmap, label in [
        ('CCI', cci_min, cci_max, 'YlGn', 'CCI'),
        ('NDVI', ndvi_min, ndvi_max, 'YlGn', 'NDVI'),
    ]:
        fig, ax = plt.subplots(figsize=(14, 8), subplot_kw={'projection': ccrs.PlateCarree()})
        
        im = ax.pcolormesh(ds.lon, ds.lat, ds[var],
                           transform=ccrs.PlateCarree(),
                           cmap=cmap, vmin=vmin, vmax=vmax)
        
        ax.coastlines(resolution='50m', linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor='gray')
        
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02, fraction=0.046)
        cbar.set_label(label, fontsize=11)
        
        ax.set_title(f'MCD19A1 {MODIS_tile} - {label} - DOY bin {doy_str}', fontsize=13)
        
        out_path = os.path.join(out_dir, f'{MODIS_tile}_{var}_doy{doy_str}.png')
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out_path}")
    
    ds.close()

print("Done.")