#!/bin/bash
#SBATCH --job-name=cnn_train
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH -o HPC_logs/cnn_train.out
#SBATCH -e HPC_logs/cnn_train.err
#SBATCH --account=default

echo "Starting job on $(hostname)"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"

module load languages/python/3.12.3
module load cuda/11.8

source ~/torch_env/bin/activate
cd /user/home/sm22599/Testing/Group
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

python HPC.py
