#!/bin/bash

# MODIFY BELOW - CHANGE THE BASE JOB NAME AND INPUT ECOREGIONS TO SUBMIT
# JOB NAME WILL BE "JOBBASE_ECOREGION", e.g. "LmfillMinusBasicfillHist_Boreal"
# UTILDIR="/uufs/chpc.utah.edu/common/home/lin-group19/ltk/bashutil"
BASEDIR="/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/code/compare_Photospec_MCD19_etc/02_gapfill_compare"
LOG_BASEDIR="/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/log/LmfillMinusBasicfillHist"
JOBBASE="LmfillMinusBasicfillHist"

QC_FLAG="CloudFree_LowAOD_ClearAdj" 

# underscores are converted back to spaces inside the python script
ECOREGIONS=(
"Boreal"
"Eastern"
"Western"
"Southwest_US_Mexico"
"GREAT_PLAINS"
"NORTH_AMERICAN_DESERTS"
)


njobs=0

for ECOREGION in "${ECOREGIONS[@]}"
do
    LOGDIR="${LOG_BASEDIR}/${ECOREGION}"
    mkdir -p ${LOGDIR}/archive/ # create log dir and an archive dir to put old files

    # Define the job name
    JOBNAME="${JOBBASE}_${ECOREGION}"

    
    # archive past log files tagged with this JOBBASE
    
    CUR_JOBBASE_FILES=$(find ${LOGDIR} -maxdepth 1 -type f -name "*${JOBNAME}*") # list existing log files tagged with this JOBBASE
    if [ ! -z "${CUR_JOBBASE_FILES}" ] # if there are any existing...
    then
        mv ${LOGDIR}/*${JOBNAME}* ${LOGDIR}/archive/ # move to the 'archive' dir
    fi

    # find other files older than 2 weeks
    OLD_FILES=$(find ${LOGDIR} -maxdepth 1 -type f -mtime +14) 
    if [ ! -z "${OLD_FILES}" ] # there are any old files....
    then
        for OLD_FILE in ${OLD_FILES}
        do
        mv ${OLD_FILE} ${LOGDIR}/archive/ # move old file to the 'archive' dir
        done
    fi

    # find old status report files and delete them
    rm -f ${LOGDIR}/RUN_STATUS*.txt

    # submit job to CHPC for this ecoregion
    sbatch -D $LOGDIR --job-name=$JOBNAME --export=LOGDIR=${LOGDIR},ECOREGION=${ECOREGION},QC_FLAG=${QC_FLAG},JOBNAME=${JOBNAME} $BASEDIR/05~pyjob.sbatch
    njobs=$((njobs+1))
done

echo "Submitted ${njobs} sbatch jobs"