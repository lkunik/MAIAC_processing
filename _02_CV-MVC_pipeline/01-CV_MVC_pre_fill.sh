#!/bin/bash

# Inputs
# ${LOGDIR} ${BASEDIR} ${MODIS_TILE} ${COMPOSITE} ${QC_FLAG} ${VZA_MIN} ${VZA_MAX} ${INPUT_YEARS} ${SAVE_ORBIT_TIME}
LOGDIR=$1
BASEDIR=$2
MODIS_TILE=$3
COMP=$4
QC_FLAG=$5
VZA_MIN=$6
VZA_MAX=$7
INPUT_YEARS_ENCODED=$8
SAVE_ORBIT_TIME=${9:-0} # 1 = save per-pixel orbit UTC/local solar time diagnostic in the composites

if [ "${SAVE_ORBIT_TIME}" == "1" ]; then
    ORBIT_TIME_FLAG="--save-orbit-time"
else
    ORBIT_TIME_FLAG=""
fi

# Decode underscore-separated years back to array
YEARS_ARRAY=(${INPUT_YEARS_ENCODED//_/ })  # Replace underscores with spaces

echo "========================================"
echo "Step 1: CV-MVC Pre-fill Composites"
echo "========================================"
echo "Tile: ${MODIS_TILE}"
echo "Composite: ${COMP}"
echo "QC Flag: ${QC_FLAG}"
echo "Years: ${YEARS_ARRAY[@]}"
echo "Save orbit time: ${SAVE_ORBIT_TIME}"
echo "Job started at: $(date)"
echo ""


# set data and working directories
# Note that the variables MODIS_TILE, YYYY, COMP, QC_FLAG, and JOBBASE come from the parent script's call to this script
export BASEDIR=/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI
export WORKINGDIR=${BASEDIR}/code/_02_CV-MVC_pipeline
export TIMESTAMP_SCRIPT=/uufs/chpc.utah.edu/common/home/lin-group19/ltk/bashutil/10_print_datetime_info.sh

# Script to run
export EXECUTION_SCRIPT="${WORKINGDIR}/01_process_MCD19A1_CV-MVC_pre-fill_composites_batch.py"


# Loop through all years
for YEAR in ${YEARS_ARRAY[@]}; do

    ### run the program(s)
    $TIMESTAMP_SCRIPT
    echo "Running ${EXECUTION_SCRIPT}  $MODIS_TILE $YEAR $COMP $QC_FLAG $VZA_MIN $VZA_MAX ${ORBIT_TIME_FLAG}"
    echo ""


    # we need a specific environment other than the base env, which we will call here
    # echo "python ${EXECUTION_SCRIPT} $MODIS_TILE $YEAR $COMP $QC_FLAG" # pass the year, month, sub-month (a or b) as argument to the script
    python ${EXECUTION_SCRIPT} $MODIS_TILE $YEAR $COMP $QC_FLAG $VZA_MIN $VZA_MAX ${ORBIT_TIME_FLAG} # pass the year, month, sub-month (a or b) as argument to the script
    EXIT_STATUS=$? # get the exit status code from the last python script
    
    
    if [ ${EXIT_STATUS} -ne 0 ]; then
        echo "ERROR: Year ${YEAR} composite ${COMP} failed with exit code ${EXIT_STATUS}"
        echo "Failed at: $(date)"
        exit ${EXIT_STATUS}
    fi
    
    echo "Year ${YEAR} completed successfully at $(date)"
    echo ""
    


done

echo "========================================"
echo "All years completed for composite ${COMP}"
echo "Job finished at: $(date)"
echo "========================================"

exit 0