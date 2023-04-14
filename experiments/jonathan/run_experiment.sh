#!/bin/bash
#SBATCH --partition=iris-hi
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --mem=100G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:1

#SBATCH --job-name="Jonathan_Exp"
#SBATCH --output=/iris/u/jyang27/cluster/outputs/exp-%j.out
source ~/.bashrc
conda activate iris_env

cd /iris/u/jyang27/dev/iris_robot_learning

python experiments/jonathan/bc_image.py --buffers /iris/u/jyang27/training_data/franka_nodesired/* /iris/u/jyang27/training_data/wx250_nodesired_control3/* /iris/u/jyang27/training_data/wx250_nodesired_control3_ee2/* --cnn medium --embedding-mode None --use-bc --seed 1 --downsample-image --color-jitter --align-actions --continuous-to-blocking --pretrained-checkpoint /iris/u/jyang27/logs/23-02-24-BC-wx250/23-02-24-BC-wx250_2023_02_24_14_11_37_id000--s2/itr_1000.pt


#done

