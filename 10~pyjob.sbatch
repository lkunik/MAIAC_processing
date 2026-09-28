#!/bin/bash

#SBATCH --requeue # on guest nodes, opt to requeue any job that is booted out by a higher priority job 
#SBATCH --time=0:30:00 # walltime, abbreviated by -t
#SBATCH --nodes=1      # number of cluster nodes, abbreviated by -N
#SBATCH --ntasks=1     # number of tasks to run
#SBATCH --cpus-per-task=8 # number of CPUs to use per task, abbreviated by -c
#SBATCH -o out-%x-%j-%N.out # name of the stdout, using the job name (%x), job number (%j) and the first node (%N)
#SBATCH -e err-%x-%j-%N.err # name of the stderr, using the job name (%x), job number (%j) and the first node (%N)
#SBATCH --account=owner-guest    # account - abbreviated by -A
#SBATCH --partition=notchpeak-guest  # partition, abbreviated by -p
#SBATCH --mem=100G

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

# set data and working directories
# Note that the variables MODIS_TILE, YYYY, CP, QC_FLAG, and JOBBASE come from the parent script's call to this script
export BASEDIR=/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI
export WORKINGDIR=${BASEDIR}/code
export TIMESTAMP_SCRIPT=/uufs/chpc.utah.edu/common/home/lin-group19/ltk/bashutil/10_print_datetime_info.sh
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

# Script to run
export EXECUTION_SCRIPT="${WORKINGDIR}/10_extract_MCD19_TerraClimate_StdAnom_vals_batch.py"


### run the program(s)
$TIMESTAMP_SCRIPT
echo "Running ${EXECUTION_SCRIPT}"
echo ""


# we need a specific environment other than the base env, which we will call here
python ${EXECUTION_SCRIPT} $ECOREGION $AVG_WINDOW $QC_FLAG # pass the names of the climate and vegetation variables as arguments to the script
EXIT_STATUS=$? # get the exit status code from the last python script
$TIMESTAMP_SCRIPT

PRINT_EXIT_STATUS_TO_FILE ${EXIT_STATUS} ${EXECUTION_SCRIPT} ${ALL_SUCCESS}
ALL_SUCCESS=$? # get the exit status code from the function


# At the end of the batch run, change the filename of the "RUN_STATUS" file to either SUCCESS_ or FAIL_
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
if [ $ALL_SUCCESS -eq 0 ]; then
   mv ${LOGFILE} ${LOGDIR}/RUN_STATUS_SUCCESS_${JOBNAME}_${TIMESTAMP}.txt
else
   mv ${LOGFILE} ${LOGDIR}/RUN_STATUS_FAIL_${JOBNAME}_${TIMESTAMP}.txt
fi
