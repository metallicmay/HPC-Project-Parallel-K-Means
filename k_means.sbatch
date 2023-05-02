#!/bin/bash 
#SBATCH --nodes=5                       # requests 10 compute servers
#SBATCH --ntasks-per-node=1              # runs 4 tasks on each server
#SBATCH --cpus-per-task=1                # uses 1 compute core per task
#SBATCH --time=2:00:00
#SBATCH --mem=2GB
#SBATCH --job-name=k_means
#SBATCH --mail-type=END
#SBATCH --mail-user=xh2223@nyu.edu
#SBATCH --output=k_means.out

mpirun -np 5 ./k_means 5