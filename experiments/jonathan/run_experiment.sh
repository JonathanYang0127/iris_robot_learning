#!/bin/bash
#SBATCH --partition=iris-hi
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --mem=50G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:1

#SBATCH --job-name="Jonathan Exp"
#SBATCH --output=/iris/u/jyang27/cluster/outputs/sample-%j.out
source ~/.bashrc
conda activate iris_env
cd /iris/u/jyang27/dev/iris_robot_learning

python experiments/jonathan/bc_image.py --buffers /iris/u/jyang27/training_data/purple_marker_grasp/combined_trajectories.npy --use-robot-state --cnn medium --embedding-mode one-hot --use-bc --seed 0

# done
