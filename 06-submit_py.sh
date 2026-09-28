#!/bin/bash

# MODIFY BELOW - CHANGE THE BASE JOB NAME AND INPUT YEARS TO SUBMIT
# JOB NAME WILL BE "JOBBASE_YYYY", e.g. "pyjob_2021"
# UTILDIR="/uufs/chpc.utah.edu/common/home/lin-group19/ltk/bashutil"
BASEDIR="/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI"
LOG_BASEDIR="${BASEDIR}/log/ecoregion_seasonality"
QC_FLAG="CloudFree_LowAOD_ClearAdj"
JOBBASE="Seasonality"
LC_threshold_high=0.99
LC_threshold_low=0.2

ECOREGIONS=(
# "Boreal"
"Eastern"
# "Western"
# "Southwest_US_Mexico"
# "GREAT_PLAINS"
# "NORTH_AMERICAN_DESERTS"
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

    # submit job to CHPC for this polygon
    sbatch -D $LOGDIR --job-name=$JOBNAME --export=LOGDIR=${LOGDIR},ECOREGION=${ECOREGION},QC_FLAG=${QC_FLAG},LC_threshold_high=${LC_threshold_high},LC_threshold_low=${LC_threshold_low},JOBNAME=${JOBNAME} $BASEDIR/code/06~pyjob.sbatch
    njobs=$((njobs+1))

done

echo "Submitted ${njobs} sbatch jobs"
