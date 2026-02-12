#!/bin/bash

#SBATCH --job-name=PG_aligner
#SBATCH --output=slurm_logs/PG_aligner_%j.out
#SBATCH --error=slurm_logs/PG_aligner_%j.err
#SBATCH --time=2:00:00
#SABTCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=80
#SBATCH --qos acc_debug
#SBATCH -N1
#SBATCH --account bsc88
##SBATCH --qos=acc_debug
umask 007
source /gpfs/scratch/bsc88/bsc088400/MT-aligner-pipeline/venv_multiprocessing_26_11_25/bin/activate

python run_pipeline.py --config configs/apdcat.yaml