#!/bin/bash
# This is the SLURM submission script

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=1:00:00

# set name of job
#SBATCH --job-name=bandits

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=jan.leike@philosophy.ox.ac.uk

cores='16'

# load the environment
module load python/anaconda2

# run the application 16 times
for i in `seq 1 $cores`; do
    python "$1" "$i" &
done
