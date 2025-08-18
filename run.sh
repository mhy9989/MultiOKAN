#!/bin/bash
#SBATCH -J kan
#SBATCH --gpus=1
#SBATCH -o slurm-%j.log
#SBATCH -e slurm-%j.log

source ~/.bashrc
conda activate mhypy
OUTPUT_LOG=train_script.log

MASTER_PORT=$((21000 + RANDOM % 34536)) 

deepspeed --master_port=$MASTER_PORT main.py


  
