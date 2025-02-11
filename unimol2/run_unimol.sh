#!/bin/bash

#PBS -q gpuvolta
#PBS -P um09
#PBS -l ncpus=48
#PBS -l ngpus=4
#PBS -l mem=128GB
#PBS -l walltime=35:00:00
#PBS -l storage=scratch/um09
#PBS -l jobfs=100GB

cd /scratch/um09/hl4138
module load python3/3.10.4 cuda/11.7
module list

source unimol-venv/bin/activate
cd Uni-Mol/unimol2

./finetune.sh