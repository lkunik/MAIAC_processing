#!/bin/bash

# Inputs
# ${LOGDIR} ${BASEDIR} ${MODIS_TILE} ${COMPOSITE} ${QC_FLAG}
LOGDIR=$1
BASEDIR=$2
MODIS_TILE=$3
COMP=$4
QC_FLAG=$5

echo "========================================"
echo "Step 2: CV-MVC Calculate Historical Averages"
echo "========================================"
echo "Tile: ${MODIS_TILE}"
echo "Composite: ${COMP}"
echo "QC Flag: ${QC_FLAG}"
echo "Job started at: $(date)"
echo ""

# set data and working directories
# Note that the variables MODIS_TILE, YYYY, CP, QC_FLAG, and JOBBASE come from the parent script's call to this script
export BASEDIR=/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI
export WORKINGDIR=${BASEDIR}/code/_04_HighVZA-MVC_pipeline
export TIMESTAMP_SCRIPT=/uufs/chpc.utah.edu/common/home/lin-group19/ltk/bashutil/10_print_datetime_info.sh

# Script to run
export EXECUTION_SCRIPT="${WORKINGDIR}/02_process_historic_avg_composites.py"
# export EXECUTION_SCRIPT="${WORKINGDIR}/02b_process_historic_avg_composites_FilterOutliers.py"

### run the program(s)
$TIMESTAMP_SCRIPT
echo "Running ${EXECUTION_SCRIPT} $MODIS_TILE $COMP $QC_FLAG"
echo ""


# we need a specific environment other than the base env, which we will call here
# echo "python ${EXECUTION_SCRIPT} $MODIS_TILE $COMP $QC_FLAG" # pass the year, month, sub-month (a or b) as argument to the script
python ${EXECUTION_SCRIPT} $MODIS_TILE $COMP $QC_FLAG # pass the year, month, sub-month (a or b) as argument to the script
EXIT_STATUS=$? # get the exit status code from the last python script


if [ ${EXIT_STATUS} -ne 0 ]; then
    echo "ERROR: composite ${COMP} failed with exit code ${EXIT_STATUS}"
    echo "Failed at: $(date)"
    exit ${EXIT_STATUS}
fi

echo "Composite ${COMP} completed successfully at $(date)"
echo ""

$TIMESTAMP_SCRIPT

exit 0