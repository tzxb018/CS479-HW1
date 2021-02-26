#!/bin/sh
#SBATCH --time=6:00:00          # Maximum run time in hh:mm:ss
#SBATCH --mem=16000             # Maximum memory required (in megabytes)
#SBATCH --job-name=479_fmnist  # Job name (to track progress)
#SBATCH --partition=cse479      # Partition on which to run job
#SBATCH --gres=gpu:1            # Don't change this, it requests a GPU

module load anaconda
conda activate tensorflow-env

python ./main-FMNIST.py > ./fmnist_output/fmnist_output.txt
