This is the main processing pipeline for the Constrained View-angle - Maximum Value
Composite  and OLS model gap-filling algorithm

all batch runs are arranged and submitted via the 00* scripts. These list the order
of all the steps here. Steps marked (RUN OUTSIDE) are not called by 00~CV-MVC_pipeline.sbatch
and must be submitted separately with their own NN-submit_py.sh script, once the steps they
depend on have finished for ALL composite periods. 00a_check_pipeline_file_completion.py checks
the outputs of each step in order.

For the purpose of the manuscript, we only need to document steps 01, 02, 03, 04, 07, 08 and 11
(all the way to 1km -> 0.01° regridding). Steps marked (QA) only produce quality-assurance
diagnostics / inputs for the figures in ../compare_Photospec_MCD19_etc/02_gapfill_compare/ and are
not needed to produce the final data.

Note that any scripts with filenames that end in "_test.py" are just for my debugging and testing purposes.
those can be ignored.

Step order summary:
  01  CV-MVC pre-fill composites                        -> pre-fill/
  02  historical climatology of pre-fill composites     -> pre-fill/{tile}/hist_avg/
  03  outlier filter (RUN OUTSIDE)                      -> pre-fill_OutlierFilter/, QC_OutlierFilter/
  04  historical average of outlier-filtered data       -> pre-fill_OutlierFilter/{tile}/hist_avg/
  05  annualize outlier filter counts (QA, RUN OUTSIDE) -> QC_OutlierFilter/{tile}/annualized/
  06  difference from historical median (QA, RUN OUTSIDE) -> QC_OutlierFilter/{tile}/diff_from_median/
  07  gap-fill scaling factors                          -> gapfill_scale/
  08  apply gap-fill (OLS model)                        -> lm-fill/
  09  gap-fill QC diagnostics (QA, RUN OUTSIDE)         -> QC_gapfill/
  10  annualize gap-fill flag counts (QA, RUN OUTSIDE)  -> QC_gapfill/{tile}/annualized/
  11  regrid 1km -> 0.01°                               -> 01d_rect/
  12  historical average of gap-filled data (1km)       -> lm-fill/{tile}/hist_avg/
  13  historical average of gap-filled data (0.01°)     -> 01d_rect/{tile}/hist_avg/
  14  summertime (JJA) averages, 0.01° (RUN OUTSIDE)
  15  standardized anomalies (1km)
  16  regrid standardized anomalies to 0.01°
  17  summertime (JJA) averages of standardized anomalies (RUN OUTSIDE)
  18  summertime (JJA) averages, 1km (RUN OUTSIDE)

step 01:
After MODIS tiles have been downloaded (see previous dir's code _01_download_pipeline)
then 16-day composites are made from the daily L2 files based on the CV-MVC algorithm
(there are various publications/sources which describe this - can infer based on the
code or recycle language from other pubs). Data are also filtered for
necessary QC fields based on the Status_QA field of the L2 files. Namely, we filter
out pixels which are adjacent to cloudy or snowy pixels. We also filter out pixels
with high AOD retrievals. And, we develop a QC flag here which includes the snow/ice identification
The data are saved in an output folder in lingroup28 called "pre-fill" because they have yet to be gap-filled

Note that there is an option to specify min and max VZA in the inputs to the script,
and this is because there are some tests where we constrained the view angle in 15-deg increments
to see how they compared. These were just used to generate "pre-fill" data for those
VZA-constrained CV-MVC comparisons.

step 02:
after all 26 years of 16-day composite data have been processed, compile all of those
together on a per-time-of-year basis, using the 23 annual composites as the time of year
For each composite, gather all 26 years of "pre-fill" data and compute the historical
climatology statistics per pixel (mean, median, median absolute deviation (MAD), standard
deviation, and number of valid years). Pixels with fewer than 4 valid years are set to NaN.
These are saved to the pre-fill/{tile}/hist_avg/ dir and are used by step 03 for outlier detection.

step 03 (RUN OUTSIDE - 03-submit_py.sh, per tile-year):
perform outlier detection using a median absolute deviation technique on the NDVI and CCI data.
A value is flagged if it is more than k * MAD away from the historical median for its composite
period (k = 3, median and MAD from step 02). See the script for how runs of consecutive
exceedances are handled. A copy of the "pre-fill" data with outliers
filtered is saved in a directory called "pre-fill_OutlierFilter", and the annual number of
outliers per pixel is saved in "QC_OutlierFilter"

step 04:
the outlier-filtered data from step 03 are averaged across all 26 years and saved to the
pre-fill_OutlierFilter/{tile}/hist_avg/ dir. This is the historical average used for gap-filling.

step 05 (QA, RUN OUTSIDE - 05-submit_py.sh, per tile):
summarizes the per-year QC_OutlierFilter files from step 03 into one file per tile with the mean
annual fraction of composites flagged as outliers at each pixel.
Used for the outlier maps in ../compare_Photospec_MCD19_etc/02_gapfill_compare/02_*

step 06 (QA, RUN OUTSIDE - 06-submit_py.sh, per tile-year, 2001-2024):
computes the difference of each composite's CCI and NDVI from the historical median (step 02),
saved alongside the outlier masks from step 03.
Used for the ecoregion 2D histograms in ../compare_Photospec_MCD19_etc/02_gapfill_compare/03_*

step 07:
this is a precursor to the gap-filling step, where after the historical average has been
saved, we load in the outlier-filtered data as well as some auxiliary datasets with information
about the biogeographic charactaristics of each pixel. These are - gridded masks of
each EPA level 2 ecoregion in north america (to group similar-functioning areas),
an elevation dataset USGS GTOPO30, and  fractional coverage of 3 broad land cover
categories (Forest, Crop and other vegetation) from the ESA-CCI initiative
Here, basically we just make a dataframe for each pixel per year per composite, and this dataframe has the
identifying information about the pixel (elevation, fractional land cover for the 3 types)
as well as a scale factor. The scale factor is the ratio of that pixel's value relative to
the historical average for each of the surface reflectance values (or uncertainties)
needed to compute the VIs. A csv file is saved per ecoregion per tile per year per composite period
to be compiled in the next step

step 08:
this is the step where all the valid scaling factors from the previous step are loaded as
input data and used to generate an OLS model that operates on a per-ecoregion basis.
It generates predicted values for any missing data that was either pre-filtered (already missing/nan)
in the original L2 files (e.g. flagged as cloudy or snowy or bright) or filtered out
by either the QC step or outlier ID step. Note that it has a sanity check on the gap-filling of
the data in the form of this standardized distance matrix and a filter using "leverage" which
is a term that needs to be explained in the manuscript. Basically, the top 1% most extrapolated values
are removed in order to prevent gapfilling from poorly-constrained limits of model space.
Also note that this is the stage at which uncertainty from Surface reflectance bands 1 and 2 are used
for propagation to determine an uncertainty value for NDVI and for CCI.
An aside that is not to be added formally to the manuscript - but I don't have a great feeling about using
this scaling/gap-filling approach to assign uncertainty to gap-filled NDVI and CCI. That will go in a
comment on the document when I send to folks.

step 09 (QA, RUN OUTSIDE - 09-submit_py.sh, per tile-year):
per-year gap-filling diagnostics. For each composite, flags which pixels were filled by the
OLS model (lm-fill, step 08) vs. kept from the original pre-fill data, and computes lm-fill and
basic-fill minus the historical mean for NDVI and CCI. Saved to "QC_gapfill".
NOTE: requires both step 08 (lm-fill) AND the basic-fill output from ../_03_basic_gapfill_pipeline/
Used for the ecoregion 2D histograms in ../compare_Photospec_MCD19_etc/02_gapfill_compare/05_*

step 10 (QA, RUN OUTSIDE - 10-submit_py.sh, per tile):
summarizes the per-year QC_gapfill files from step 09 into one file per tile with the mean annual
number (and fraction) of composites that were gap-filled at each pixel.
Used for the gap-fill maps in ../compare_Photospec_MCD19_etc/02_gapfill_compare/04_*

Step 11:
This step uses the xESMF package to regrid the data from MODIS sinusoidal projection (which
uses 2-dimensional x and y coordinates) to a rectilinear lon/lat gridded projection.
