import xarray as xr, numpy as np, sys
B='/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/'
Q='CloudFree_LowAOD_ClearAdj'
for t in sys.argv[1:]:
    a=xr.open_dataset(f'{B}QC_gapfill/QCfilt_{Q}/{t}/annualized/MCD19A1.{t}.061_1km_annual_CV-MVC_QCfilt_{Q}_gapfillQC.nc')
    v=a.gapfill_CCI_frac.values
    print(t,'annual frac: mean',np.nanmean(v),'max',np.nanmax(v),'frac>0',np.mean(v>0))
    d=xr.open_dataset(f'{B}QC_gapfill/QCfilt_{Q}/{t}/MCD19A1.{t}.061_1km_16day_CV-MVC_gapfillQC_QCfilt_{Q}_2017.nc')
    g=d.gapfill_CCI
    print(' 2017 per period: n==1', [int((g.sel(composite_period=p)==1).sum()) for p in range(23)])
    print(' 2017 per period: n==0', [int((g.sel(composite_period=p)==0).sum()) for p in range(23)])
