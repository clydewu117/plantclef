#!/usr/bin/env bash
#SBATCH --job-name=conv
#SBATCH --account=PAS2099
#SBATCH --partition=quad
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xu.5755@osu.edu
#SBATCH --output=/fs/scratch/PAS2099/plantclef/logs/slurm-%j-conv.out

module load miniconda3/24.1.2-py310

conda activate plantclef

python train_convnext.py
