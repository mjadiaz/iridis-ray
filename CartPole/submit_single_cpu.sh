#!/bin/bash 
#SBATCH --job-name=RLlibPheno
#SBATCH --time=06:00:00
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=40

source /home/mjad1g20/.bashrc
source activate rrlib

#python simple_trainer_exp.py --num-cpus 39
python simple_trainer_ppo.py --num-cpus 39

