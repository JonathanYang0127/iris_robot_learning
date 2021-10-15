s = """sbatch -A fc_rail -p savio2 -t 720 -N 1 -n 1 --cpus-per-task=1 --time 1440 --wrap=$'singularity exec --writable -B /usr/lib64 -B /var/lib/dcv-gl -B /global /global/home/groups/co_rail/anair17/railrl_hand_v2 /bin/bash -c \\'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/global/home/users/anair17/.mujoco/mujoco200/bin;source /global/software/sl-7.x86_64/modules/langs/python/3.7/etc/profile.d/conda.sh;conda activate /global/home/groups/co_rail/anair17/python/tfagentsenv;export D4RL_DATASET_DIR=/global/scratch/users/anair17/s3doodad/d4rl_data;export PYTHONPATH=/global/scratch/users/anair17/s3doodad/mj_envs:$PYTHONPATH;export PYTHONPATH=/global/scratch/users/anair17/s3doodad/mjrl:$PYTHONPATH;export PYTHONPATH=/global/scratch/users/anair17/s3doodad/hand_dapg:$PYTHONPATH;export PYTHONPATH=/global/scratch/users/anair17/s3doodad/bullet-objects:$PYTHONPATH;export PYTHONPATH=/global/scratch/users/anair17/s3doodad/bullet-manipulation:$PYTHONPATH;export PYTHONPATH=/global/scratch/users/anair17/s3doodad/metaworld:$PYTHONPATH;export PYTHONPATH=/global/scratch/users/anair17/s3doodad/agents:$PYTHONPATH;export PYTHONPATH=/global/scratch/users/anair17/s3doodad/d4rl:$PYTHONPATH;export TFAGENTS_D4RL_DATA=/global/scratch/users/anair17/s3doodad/d4rl_tf_datasets;export TFAGENTS_ROOT_DIR=/global/scratch/users/anair17/s3doodad/cql_sac;export PYTHONPATH=$PYTHONPATH:/global/scratch/users/anair17/s3doodad/railrl-private:/global/scratch/users/anair17/s3doodad/multiworld:/global/scratch/users/anair17/s3doodad/doodad:/global/scratch/users/anair17/s3doodad/d4rl:/global/scratch/users/anair17/s3doodad/mj_envs:/global/scratch/users/anair17/s3doodad/mjrl:/global/scratch/users/anair17/s3doodad/hand_dapg;cd /global/scratch/users/anair17/s3doodad/railrl-private;python experiments/ashvin/awacp/tfcql/cql_sbatch_online1.py --env_name %s --seed %d\\''"""

with open("/tmp/script_to_scp_over.sh", "w") as myfile:
    for env_name in ["antmaze-umaze-v0", "antmaze-umaze-diverse-v0", "antmaze-medium-play-v0",
        "antmaze-medium-diverse-v0", "antmaze-large-diverse-v0", "antmaze-large-play-v0",
    ]:
        for seed in range(5):
            cmd = s % (env_name, seed)
            myfile.write(cmd + '\n')
            myfile.write('sleep 10\n')
