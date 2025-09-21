#!/bin/bash 

#SBATCH --time=168:00:00
#SBATCH --account=st-cdeboer-1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --mem=4G
#SBATCH --job-name=testK
#SBATCH -e slurm-%A_%a.err   # %A = main job ID, %a = array index
#SBATCH -o slurm-%A_%a.out
#SBATCH --mail-user=chapelm@student.ubc.ca
#SBATCH --mail-type=ALL
#SBATCH --array=0-9          # run replicate indices 0 through 9

# Load software environment 
module load miniconda3

# Activate conda environment 
source activate grn-evo

cd $SLURM_SUBMIT_DIR

# Use the array index as the replicate number
python evolution.py testK rep${SLURM_ARRAY_TASK_ID}
