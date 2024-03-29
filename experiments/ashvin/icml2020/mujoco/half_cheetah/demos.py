from rlkit.demos.collect_demo import collect_demos_fixed
from rlkit.util.io import load_local_or_remote_file

from rlkit.launchers.experiments.awac.awac_rl import ENV_PARAMS

if __name__ == '__main__':
    data = load_local_or_remote_file('ashvin/icml2020/mujoco/reference/run1/id0/itr_1000.pkl')
    env = data['evaluation/env']
    policy = data['exploration/policy']
    policy.to("cpu")
    env_name = "half-cheetah"
    outfile = "/home/ashvin/data/s3doodad/demos/icml2020/mujoco/%s.npy" % env_name
    horizon = ENV_PARAMS[env_name]['max_path_length']
    collect_demos_fixed(env, policy, outfile, N=100, horizon=horizon) # , threshold=.1, add_action_noise=False, key='puck_distance', render=True, noise_sigma=0.0)
