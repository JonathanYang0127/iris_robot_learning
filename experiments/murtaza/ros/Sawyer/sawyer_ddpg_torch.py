import random

from railrl.envs.wrappers import convert_gym_space
from railrl.exploration_strategies.base import PolicyWrappedWithExplorationStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.launchers.launcher_util import continue_experiment
from railrl.launchers.launcher_util import resume_torch_algorithm
from railrl.torch.networks import FeedForwardQFunction, FeedForwardPolicy
from railrl.torch.ddpg import DDPG
from railrl.envs.ros.sawyer_env import SawyerEnv
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.torch import pytorch_util as ptu
import sys

def example(variant):
    env_class = variant['env_class']
    env_params = variant['env_params']
    env = env_class(**env_params)
    obs_space = convert_gym_space(env.observation_space)
    action_space = convert_gym_space(env.action_space)
    es_class = variant['es_class']
    es_params = dict(
        action_space=action_space,
        **variant['es_params']
    )
    use_gpu = variant['use_gpu']
    es = es_class(**es_params)
    qf = FeedForwardQFunction(
        int(env.observation_space.flat_dim),
        int(env.action_space.flat_dim),
        100,
        100,
    )
    policy_class = variant['policy_class']
    policy_params = dict(
        obs_dim=int(obs_space.flat_dim),
        action_dim=int(action_space.flat_dim),
        fc1_size=100,
        fc2_size=100,
    )
    policy = policy_class(**policy_params)
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    algorithm = DDPG(
        env,
        qf=qf,
        policy=policy,
        exploration_policy=exploration_policy,
        **variant['algo_params'],
    )
    if use_gpu and ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()

experiments=[
    'joint_angle|fixed_angle',
    'joint_angle|varying_angle',
    'end_effector_position|fixed_ee',
    'end_effector_position|varying_ee',
    'end_effector_position_orientation|fixed_ee',
    'end_effector_position_orientation|varying_ee'
]

max_path_length = 100
if __name__ == "__main__":
    try:
        exp_dir = sys.argv[1]
    except:
        exp_dir = None
    if exp_dir:
        continue_experiment(exp_dir, resume_function=resume_torch_algorithm)
    else:
        for i in range(3):
            run_experiment(
                example,
                exp_prefix="ddpg-sawyer-varying-end-effector-NIPS",
                seed=random.randint(0, 666),
                mode='local',
                variant={
                    'version': 'Original',
                    'max_path_length': max_path_length,
                    'use_gpu': True,
                    'es_class': OUStrategy,
                    'env_class': SawyerEnv,
                    'policy_class': FeedForwardPolicy,
                    'normalize_env': False,
                    'env_params': {
                        'arm_name': 'right',
                        'safety_box': True,
                        'loss': 'huber',
                        'huber_delta': 10,
                        'safety_force_magnitude': 8,
                        'temp': 15,
                        'remove_action': False,
                        'experiment': experiments[3],
                        'reward_magnitude': 1,
                        'use_safety_checks':False,
                    },
                    'es_params': {
                        'max_sigma': .25,
                        'min_sigma': .25,
                    },
                    'algo_params': dict(
                        batch_size=64,
                        num_epochs=50,
                        num_updates_per_env_step=3,
                        num_steps_per_epoch=1000,
                        max_path_length=max_path_length,
                        num_steps_per_eval=1000,
                    ),
                },
                use_gpu=True,
            )
