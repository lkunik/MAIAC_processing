#!/bin/bash

# Inputs
# ${LOGDIR} ${BASEDIR} ${MODIS_TILE} ${COMPOSITE} ${QC_FLAG} ${INPUT_YEARS}
LOGDIR=$1
BASEDIR=$2
MODIS_TILE=$3
COMP=$4
QC_FLAG=$5
INPUT_YEARS_ENCODED=$6

# Decode underscore-separated years back to array
YEARS_ARRAY=(${INPUT_YEARS_ENCODED//_/ })  # Replace underscores with spaces

echo "========================================"
echo "Step 15: CV-MVC Calculate Standardized Anomalies from Gap-Fill Composites and Historical Averages"
echo "========================================"
echo "Tile: ${MODIS_TILE}"
echo "Composite: ${COMP}"
echo "QC Flag: ${QC_FLAG}"
echo "Years: ${YEARS_ARRAY[@]}"
echo "Job started at: $(date)"
echo ""

# set data and working directories
# Note that the variables MODIS_TILE, YYYY, CP, QC_FLAG, and JOBBASE come from the parent script's call to this script
export BASEDIR=/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI
export WORKINGDIR=${BASEDIR}/code/_02_CV-MVC_pipeline
export TIMESTAMP_SCRIPT=/uufs/chpc.utah.edu/common/home/lin-group19/ltk/bashutil/10_print_datetime_info.sh

# Script to run
export EXECUTION_SCRIPT="${WORKINGDIR}/15_process_std_anomaly_gapfill_composites_1km_curv.py"


# Loop through all years
for YEAR in ${YEARS_ARRAY[@]}; do

    ### run the program(s)
    $TIMESTAMP_SCRIPT
    echo "Running ${EXECUTION_SCRIPT} $MODIS_TILE $YEAR $COMP $QC_FLAG"
    echo ""


    # we need a specific environment other than the base env, which we will call here
    # echo "python ${EXECUTION_SCRIPT} $MODIS_TILE $YEAR $COMP $QC_FLAG" # pass the year, month, sub-month (a or b) as argument to the script
    python ${EXECUTION_SCRIPT} $MODIS_TILE $YEAR $COMP $QC_FLAG # pass the year, month, sub-month (a or b) as argument to the script
    EXIT_STATUS=$? # get the exit status code from the last python script
    
    
    if [ ${EXIT_STATUS} -ne 0 ]; then
        echo "ERROR: Year ${YEAR} composite ${COMP} failed with exit code ${EXIT_STATUS}"
        echo "Failed at: $(date)"
        continue
    fi
    
    echo "Year ${YEAR} completed successfully at $(date)"
    echo ""
    $TIMESTAMP_SCRIPT


done

echo "========================================"
echo "All years completed for composite ${COMP}"
echo "Job finished at: $(date)"
echo "========================================"

exit 0