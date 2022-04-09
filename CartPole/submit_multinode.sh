#!/bin/bash 

#SBATCH --job-name=cartpole-multiple-nodes
#SBATCH --time=00:20:00
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --nodes=4

source /home/mjad1g20/.bashrc
source activate rrlib

ip_prefix=$(srun --nodes=1 --ntasks=1 -w $node1 hostname --ip-address)
port=6379
ip_head=$ip_prefix:$port
redis_password=$(uuidgen)


