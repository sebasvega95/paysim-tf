#!/bin/bash
#SBATCH --output=res_paysim.txt
#SBATCH --nodes=3
#SBATCH --gres=gpu:1

srun python3 main.py

