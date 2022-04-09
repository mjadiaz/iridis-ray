#!/bin/bash 
#SBATCH -p gtx1080
#SBATCH --gres=gpu:1
#SBATCH --job-name=RLlibPheno
#SBATCH --time=06:00:00
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=20

source /home/mjad1g20/.bashrc
source activate rrlib

python simple_trainer_exp.py --num-cpus 19 --num-gpus 1


