#!/bin/sh
#SBATCH --time=6:00:00          # Maximum run time in hh:mm:ss
#SBATCH --mem=32000             # Maximum memory required (in megabytes)
#SBATCH --job-name=479_cifar  # Job name (to track progress)
#SBATCH --partition=cse479      # Partition on which to run job
#SBATCH --gres=gpu:1            # Don't change this, it requests a GPU

module load anaconda
conda activate tensorflow-env

python ./main-CIFAR.py > ./cifar_output/CIFAR-output.txt
