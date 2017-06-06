#!/bin/bash
#SBATCH --output=res-paysim.txt
#SBATCH --nodes=3
#SBATCH --gres=gpu:1

TF_CPP_MIN_LOG_LEVEL=3 srun python3 main.py

