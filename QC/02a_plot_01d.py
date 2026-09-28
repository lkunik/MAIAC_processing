import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import glob
import random

data_dir = '/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/01d_rect/QCfilt_CloudFree/h10v04/'
out_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/plots/maps/regrid_validation/QCfilt_CloudFree/h10v04'
os.makedirs(out_dir, exist_ok=True)

files = sorted(glob.glob(os.path.join(data_dir, 'MCD19A1.h10v04.061_01d_16day_CV-MVC_QCfilt_CloudFree_*_rect.nc')))
print(f"Found {len(files)} files total")

n_sample = 20
sampled_files = random.sample(files, n_sample)
print(f"Randomly sampled {n_sample} files:")
for f in sorted(sampled_files):
    print(f"  {os.path.basename(f)}")

var_params = {
    'CCI':  (-0.3, 0.2),
    'NDVI': (0.1, 1.0),
}

for fpath in sorted(sampled_files):
    period_str = os.path.basename(fpath).replace('_rect.nc', '').split('_')[-1]

    ds = xr.open_dataset(fpath)
    ds = ds.assign(CCI=(ds["Sur_refl11"] - ds["Sur_refl1"]) / (ds["Sur_refl11"] + ds["Sur_refl1"]))
    ds = ds.assign(NDVI=(ds["Sur_refl2"] - ds["Sur_refl1"]) / (ds["Sur_refl2"] + ds["Sur_refl1"]))

    for var, (vmin, vmax) in var_params.items():
        fig, ax = plt.subplots(figsize=(14, 6), subplot_kw={'projection': ccrs.PlateCarree()})

        im = ax.pcolormesh(ds.lon, ds.lat, ds[var],
                           transform=ccrs.PlateCarree(),
                           cmap='YlGn', vmin=vmin, vmax=vmax)

        ax.coastlines(resolution='50m', linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor='gray')

        cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02, fraction=0.046)
        cbar.set_label(var, fontsize=11)

        ax.set_title(f'MCD19A1 h10v04 - {var} - {period_str}', fontsize=13)

        out_path = os.path.join(out_dir, f'h10v04_{var}_{period_str}.png')
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out_path}")

    ds.close()

print("Done.")