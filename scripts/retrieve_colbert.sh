#!/bin/sh
#SBATCH --job-name=am_colbert
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:nvidia_rtx_a6000:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:40:00 # d-h:m:s
#SBATCH --mem=64gb # memory per GPU 
#SBATCH -c 48 # number of CPUs
#SBATCH --output=logs-slurm/am_colbert_%j.out

# Set up the environment
source your/path/to/.bashrc
conda activate amharic_environment
nvidia-smi 

# Navigate to the project directory
cd /your/path/to/amharic_colbert

# Add the amharic_colbert directory to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/your/path/to/amharic_colbert
python ColBERT_AM/retrieval.py