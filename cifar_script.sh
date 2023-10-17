#!/bin/bash

#SBATCH --job-name=cifarModel
#SBATCH --account=users
#SBATCH --nodes=1
##SBATCH --nodelist=nova[32,82,83,101,102]
#SBATCH --nodelist=nova[82]
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=main
#SBATCH --gres=gpu:1
##SBATCH --mem=20G
#SBATCH --time=15-0
#SBATCH --output=cifar.out



echo "---- env ! ----"

## ulimit -s unlimited
## ulimit -l unlimited
## ulimit -a

echo "------- setup done ! -----"
## Load the python interpreter
##clear the module
# module purge
# module load cuda/11.7

## conda environment
source ${HOME}/.bashrc
eval "$(conda shell.bash hook)"

conda activate cifarTuts

srun nvidia-smi


echo "--**cifar**--"
srun python train.py 
srun python test.py 
srun python predict.py
