This directory contains the code that is used to download and process 
 MOD09 reflectance data for benchmarking/validation of the MCD19 reflectance
 data. This is used because it also contains Band 11 (and other Ocean observing bands)
reflectance data over land, which is necessary to compute the CCI. Other studies
have used this dataset to generate CCI. But the MAIAC dataset should offer improvements 
over MOD09, so here we are testing it!

The data are downloaded on a per-site basis as point extractions using python's
google earth engine package. Bands 1-7 are taken from MOD09GA v6.1 and MYD09GA v6.1 so as to 
have a combination of Aqua & Terra observations. Bands 8-12 are taken from 
MODOCGA v6.0 and MYDOCGA v6.0 (6.1 is not available on GEE). Data were only downloaded for the
periods of time for which PhotoSpec data are available. 

The CV-MVC algorithm is performed in 02_proc_MOD09_bysite_batch.py and 16-day 
composites are formulated. No gap-filling or outlier detection is done, since historical 
averages are not computed (we don't download enough years for that). Instead, we 
just compare these values to the "pre-fill" values from the MCD19 data processing
after the CV-MVC step. The code that is used to analyze that is in ../GCOM_MOD09_compare/