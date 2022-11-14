#!/bin/bash
#SBATCH --time=72:00:00 -n 1 
#SBATCH --cpus-per-task=1
#SBATCH --output slurm.out
#SBATCH --error slurm.err
#SBATCH --partition=serial
#SBATCH --job-name=gen_data


source /home/dclw/.bashrc
conda activate py39

cd /scratch/dclw/ML-DiffuseReader/main

START=$1
END=$2
origPwd=$(pwd)
 
python Generate_TrainingData.py ${START} ${END}
