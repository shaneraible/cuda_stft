#!/bin/bash

# This script will request one GPU device and 1 CPU core
#SBATCH --job-name=stft_gpu
#SBATCH --output=stft_.out
#SBATCH --error=stft_.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=a100
#SBATCH --account=ece1570_2022f

module purge
module load gcc/5.4.0
module load cuda/11.7.1

make
./stft
# ./gpu -n 20000 -o gpu.txt
