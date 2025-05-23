#!/bin/sh
#SBATCH --job-name=CS3263
#SBATCH --time=0:15:00
#SBATCH --partition=normal
#SBATCH --gres=gpu:h100-96
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
hostname 
nvidia-smi
python XLNet.py
echo -e "\nJob completed at $(date)"