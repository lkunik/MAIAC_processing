This is the main processing pipeline for the "High View-angle - Maximum Value 
Composite" and OLS model gap-filling algorithm

Important note - this code is nearly identical to the code pipeline from ../_02_CV-MVC_pipeline/ 
except that there is a modification to the compositing algorithm, where only observations
with View Zenith Angles between 40-70 degrees are considered, and of the valid 
observations after QC for AOD and adjacent clouds, the observation with the maximum 
NDVI is selected for the composite.
All other code following step 01 is basically identical to the CV-MVC code from ../_02_CV-MVC_pipeline/ 
but with separate directory locations and file identifiers.

all batch runs are arranged and submitted via the 00* scripts. These list the order 
of all the steps here.
For the purpose of the manuscript, we only need to document steps 01, 02, 03, 
04 and 05 (all the way to 1km -> 0.01° regridding)

Note that any scripts with filenames that end in "_test.py" are just for my debugging and testing purposes.
those can be ignored.

step 01:
After MODIS tiles have been downloaded (see previous dir's code _01_download_pipeline)
then 16-day composites are made from the daily L2 files based on the CV-MVC algorithm
(there are various publications/sources which describe this - can infer based on the 
code or recycle language from other pubs). Data are also filtered for 
necessary QC fields based on the Status_QA field of the L2 files. Namely, we filter 
out pixels which are adjacent to cloudy or snowy pixels. We also filter out pixels 
with high AOD retrievals. And, we develop a QC flag here which includes the snow/ice identification
The data are saved in an output folder in lingroup28 called "pre-fill" because they have yet to be gap-filled

step 02: 
after all 26 years of 16-day composite data have been processed, compile all of those 
together on a per-time-of-year basis, using the 23 annual composites as the time of year
For each composite, gather all 26 years of data, and then perform outlier detection using
a median absolute deviation technique on the NDVI and CCI data. uses k = 3 and a minimum 
of 4 values across all years of data. A copy of the "pre-fill" data with outliers 
filtered is saved in a directory called called "pre-fill_OutlierFilter"
These outlier-filtered data are then averaged across all 26 years and saved to the pre-fill/{tile}/hist_avg/ dir

step 03:
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

step 04:
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

Step 05:
This step uses the xESMF package to regrid the data from MODIS sinusoidal projection (which 
uses 2-dimensional x and y coordinates) to a rectilinear lon/lat gridded projection. 
