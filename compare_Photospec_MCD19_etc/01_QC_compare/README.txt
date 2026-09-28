The code in this directory is part of the data validation and benchmarking
steps of the MCD19 data processing.

In these scripts we compare NDVI and CCI computed from MCD19 composite reflectance
for a variety of AOD and cloud adjacency QC schemes. These comparisons are 
performed at the PhotoSpec sites and data are compared to PhotoSpec-derived VIs 
after averaging photospec data into 16-day average bins

In scripts 01*.py, the data are loaded (some of this code is commented out because 
the process of looping through the data files only had to be done once - the 
point-extracted data were then saved to temporary directory to be sourced in 
future analysis runs for that QC scheme)

in the 01*.py scripts there are linear regression analyses which compare 16-day composite values
of NDVI and CCI for the various QC schemes against 16-day averaged PhotoSpec NDVI and CCI.
We took the pearson correlation coefficient, spearman rank correlation and crmse as the major
statistics to use for benchmarking

the 02*.py, 03*.py and 04*.py scripts are just used to visualize the data analysis
and the figures will be presented in the manuscript, so these do not need to be scrutinized.

Note that in all of these comparisons with Photospec data,
the data are loaded and are filtered for the hours between 1 and 2pm local time.
For US-NR1, PhotoSpec data are additionally filtered to exclude NDVI less than 0.6.
For OSBS, PhotoSpec data are additionally filtered to exclude NDVI less than 0.5.
For Ca-Obs, PhotoSpec data are additionally filtered to exclude NDVI less than 0.4.
For DEJU, PhotoSpec data are not additionally filtered.

"Winter periods" here and in other similar post-processing comparison scripts refers to
time periods where snow cover is estimated to be present on the canopy and/or surface. These
were manually identified by looking at PhenoCam imagery for each site's canopy from visible imagery
from the flux tower.