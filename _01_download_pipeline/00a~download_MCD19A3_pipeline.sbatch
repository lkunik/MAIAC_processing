#!/bin/bash

#SBATCH --requeue # on guest nodes, opt to requeue any job that is booted out by a higher priority job 
#SBATCH --time=12:00:00 # walltime, abbreviated by -t
#SBATCH --nodes=1      # number of cluster nodes, abbreviated by -N
#SBATCH --ntasks=1     # number of tasks to run
#SBATCH --cpus-per-task=8 # number of CPUs to use per task, abbreviated by -c
# COMMENT #SBATCH -o out-%x-%j-%N.out # name of the stdout, using the job name (%x), job number (%j) and the first node (%N)
# COMMENT #SBATCH -e err-%x-%j-%N.err # name of the stderr, using the job name (%x), job number (%j) and the first node (%N)
#SBATCH --account=owner-guest    # account - abbreviated by -A
#SBATCH --partition=notchpeak-guest  # partition, abbreviated by -p
#SBATCH --mem=200G

# For lin-group kingspeak nodes, use account=lin-kp, partition=lin-kp
# For lin-group notchpeak nodes, use account=lin-np, partition=lin-np
# For guest kingspeak nodes, use account=owner-guest, partition=kingspeak-guest
# For guest notchpeak nodes, use account=owner-guest, partition=notchpeak-guest


source /uufs/chpc.utah.edu/sys/installdir/lmod/8.6-r8/init/bash
source /uufs/chpc.utah.edu/common/home/u6017162/software/pkg/miniforge3/etc/profile.d/conda.sh
conda activate main  # activate main conda environment


# load appropriate modules
module use $HOME/MyModules
module load intel
module load gcc/8.5.0
module load intel-oneapi-mpi/2021.4.0
module load gdal/3.3.3
module load mvapich2
module load miniconda3/latest



BASEDIR="/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI"
echo "Starting download pipeline for ${MODIS_TILE} year ${INPUT_YEAR}"


export TIMESTAMP_SCRIPT=/uufs/chpc.utah.edu/common/home/lin-group19/ltk/bashutil/10_print_datetime_info.sh
export STATUS_STR="${MODIS_TILE}_${INPUT_YEAR}"
export LOGFILE=${LOGDIR}/RUN_STATUS.txt
export ALL_SUCCESS=0

# Argument $1 = EXIT STATUS
# Argument $2 = EXECUTION SCRIPT
# Argument $3 = previous state of global exit statuses
PRINT_EXIT_STATUS_TO_FILE () {
  
   # Check the exit status and write to the log file
   if [ $1 -eq 0 ]; then
      echo "code ran to completion: $2" >> ${LOGFILE}
      return $3
   else
      echo "code failed, returned with exit code $1: $2" >> ${LOGFILE}
      return 1
   fi

}





# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # Step 1: Download HDF file for this tile-year
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# RUN_SCRIPT=${BASEDIR}/code/_01_download_pipeline/05_download_MCD19A3_byTile_batch.sh
# ${RUN_SCRIPT} ${MODIS_TILE} ${INPUT_YEAR}
# EXIT_STATUS=$?


# PRINT_EXIT_STATUS_TO_FILE ${EXIT_STATUS} ${EXECUTION_SCRIPT} ${ALL_SUCCESS}
# ALL_SUCCESS=$? # get the exit status code from the function

# # At the end of the batch run, change the filename of the "RUN_STATUS" file to either SUCCESS_ or FAIL_
# TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# if [ $ALL_SUCCESS -eq 0 ]; then
#     mv ${LOGFILE} ${LOGDIR}/RUN_STATUS_SUCCESS_01_${STATUS_STR}_${TIMESTAMP}.txt
# else
#     mv ${LOGFILE} ${LOGDIR}/RUN_STATUS_FAIL_01_${STATUS_STR}_${TIMESTAMP}.txt
#     exit 1 # exit the pipeline if the step fails, since subsequent steps depend on it
# fi

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Step 2: Convert HDF file to NC for this tile-year
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
RUN_SCRIPT=${BASEDIR}/code/_01_download_pipeline/06_convert_MCD19A3_hdf_to_nc_byYear_batch.sh
${RUN_SCRIPT} ${MODIS_TILE} ${INPUT_YEAR}
EXIT_STATUS=$?


PRINT_EXIT_STATUS_TO_FILE ${EXIT_STATUS} ${EXECUTION_SCRIPT} ${ALL_SUCCESS}
ALL_SUCCESS=$? # get the exit status code from the function

# At the end of the batch run, change the filename of the "RUN_STATUS" file to either SUCCESS_ or FAIL_
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
if [ $ALL_SUCCESS -eq 0 ]; then
    mv ${LOGFILE} ${LOGDIR}/RUN_STATUS_SUCCESS_02_${STATUS_STR}_${TIMESTAMP}.txt
else
    mv ${LOGFILE} ${LOGDIR}/RUN_STATUS_FAIL_02_${STATUS_STR}_${TIMESTAMP}.txt
    exit 1 # exit the pipeline if the step fails, since subsequent steps depend on it
fi


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FINAL STEPS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Save job IDs to file for monitoring
cat > ${LOGDIR}/pipeline_jobs.txt <<EOF
Pipeline for ${MODIS_TILE} year ${INPUT_YEAR}
Started: $(date)
QC Flag: ${QC_FLAG}
Years: 2000-2025 (${nYEARS} years)

Monitor: squeue -u \$USER | grep ${MODIS_TILE}
EOF

# echo ""
# echo "Pipeline submitted for ${MODIS_TILE} year ${INPUT_YEAR}"
# echo "Job chain: ${CVMVC_JOB} -> ${HIST_AVG_JOB} -> ${SCALE_JOB} -> ${APPLY_JOB} -> ${REGRID_JOB}"
# echo "Total: 5 jobs processing ${nYEARS} years each"
# echo ""
# echo "Monitor: squeue -u \$USER | grep ${MODIS_TILE}"
# echo "Details: ${LOGDIR}/pipeline_jobs.txt"