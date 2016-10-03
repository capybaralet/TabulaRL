#!/bin/bash

target='sfop0669@arcus-b.arc.ox.ac.uk'
subfolder='~/experiment2'

scp sbatch-job.sh "$target":"$subfolder"
scp ../bandits.py "$target":"$subfolder"
scp experiment.py "$target":"$subfolder"
ssh "$target" 'cd experiment2; sbatch sbatch-job.sh'
