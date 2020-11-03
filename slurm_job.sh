#!/bin/bash
#SBATCH -J w5c5-2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=1

echo "JobID: $SLURM_JOB_ID"
echo "Time: `date`"
echo "Running on node: `hostname`"
echo "Current directory: `pwd`"

module load conda

# INPUT YOUR COMMAND !! ##############
INPUT_CMD=""
######################################

python3 molfinder.py -N 100 -r $SLURM_JOB_ID  -n 300 -v 5 -c 0.5 -dc 0.90 -t "CCN1c2ccccc2Cc3c(O)ncnc13" 2> /dev/null
#wait

echo "End Time: `date`"
