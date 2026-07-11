#!/bin/bash
#SBATCH --job-name=wake
#SBATCH --output=wake.log
#SBATCH --error=wake.err
#SBATCH --time=31-00:00:00
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1M
#SBATCH --exclude=compute-0-[0-40,44]

while true; do
  sleep 10000
done