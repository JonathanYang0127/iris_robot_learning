import sys

from railrl.envs.remote import RemoteRolloutEnv
from railrl.envs.ros.sawyer_env import SawyerEnv
from railrl.envs.wrappers import convert_gym_space
from railrl.exploration_strategies.gaussian_strategy import GaussianStrategy
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import continue_experiment
from railrl.launchers.launcher_util import resume_torch_algorithm
from railrl.launchers.launcher_util import run_experiment
from railrl.policies.torch import FeedForwardPolicy
from railrl.qfunctions.torch import FeedForwardQFunction
from railrl.torch.algos.parallel_ddpg import ParallelDDPG
from rllab.envs.normalized_env import normalize
import random
import ray
import itertools


def example(variant):
    # env_class = variant['env_class']
    # env_params = variant['env_params']
    # env = env_class(**env_params)
    # obs_space = convert_gym_space(env.observation_space)
    # action_space = convert_gym_space(env.action_space)
    # es_class = variant['es_class']
    # es_params = dict(
    #     action_space=action_space,
    #     **variant['es_params']
    # )
    # policy_class = variant['policy_class']
    # use_gpu = variant['use_gpu']
    #
    # if variant['normalize_env']:
    #     env = normalize(env)
    #
    # es = es_class(**es_params)
    # qf = FeedForwardQFunction(
    #     int(env.observation_space.flat_dim),
    #     int(env.action_space.flat_dim),
    #     100,
    #     100,
    # )
    # policy_params = dict(
    #     obs_dim=int(obs_space.flat_dim),
    #     action_dim=int(action_space.flat_dim),
    #     fc1_size=100,
    #     fc2_size=100,
    # )
    # policy = policy_class(**policy_params)
    # remote_env = RemoteRolloutEnv(
    #     env_class,
    #     env_params,
    #     policy_class,
    #     policy_params,
    #     es_class,
    #     es_params,
    #     variant['max_path_length'],
    #     variant['normalize_env'],
    # )

    qf = FeedForwardQFunction(
        int(remote_env.observation_space.flat_dim),
        int(remote_env.action_space.flat_dim),
        100,
        100,
    )
    policy_class = FeedForwardPolicy
    policy_params = dict(
        obs_dim=int(remote_env.observation_space.flat_dim),
        action_dim=int(remote_env.action_space.flat_dim),
        fc1_size=100,
        fc2_size=100,
    )
    policy = policy_class(**policy_params)
    es_class = OUStrategy
    es_params = dict(
        action_space=remote_env.action_space,
        **{
            'max_sigma': .25,
            'min_sigma': .25,
        }
    )
    es = es_class(**es_params)
    use_gpu = variant['use_gpu']
    algorithm = ParallelDDPG(
        remote_env,
        exploration_strategy=es,
        qf=qf,
        policy=policy,
        **variant['algo_params'],
    )
    if use_gpu:
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
policy_learning_rates = [1e-2, 1e-3, 1e-4]
taus = [1e-2, 1e-3, 1e-4]
cart_prod = list(itertools.product(policy_learning_rates, taus))
import ipdb; ipdb.set_trace()
if __name__ == "__main__":
    ray.init()
    try:
        exp_dir = sys.argv[1]
    except:
        exp_dir = None
    env_class = SawyerEnv
    env_params = {
                    'arm_name': 'right',
                    'safety_box': True,
                    'loss': 'huber',
                    'huber_delta': 10,
                    'safety_force_magnitude': 5,
                    'temp': 1,
                    'remove_action': False,
                    'experiment': experiments[2],
                    'reward_magnitude': 10,
                }
    env = env_class(**env_params)

    policy_class = FeedForwardPolicy
    obs_space = convert_gym_space(env.observation_space)
    action_space = convert_gym_space(env.action_space)
    policy_params = dict(
        obs_dim=int(obs_space.flat_dim),
        action_dim=int(action_space.flat_dim),
        fc1_size=100,
        fc2_size=100,
    )
    policy = policy_class(**policy_params)
    es_class = OUStrategy
    es_params = dict(
        action_space=action_space,
        **{
            'max_sigma':.25,
            'min_sigma':.25,
        }
    )
    global remote_env
    remote_env = RemoteRolloutEnv(
        env_class,
        env_params,
        policy_class,
        policy_params,
        es_class,
        es_params,
        max_path_length,
        False,
    )
    # import ipdb; ipdb.set_trace()
    # for i in range(0, len(cart_prod)):
    #     run_experiment(
    #         example,
    #         exp_prefix="ddpg-parallel-sawyer-fixed-end-effector-hyper-param-search" + str(cart_prod[0]),
    #         seed=random.randint(0, 666),
    #         mode='here',
    #         variant={
    #             'version': 'Original',
    #             'max_path_length': max_path_length,
    #             'use_gpu': True,
    #             'algo_params': dict(
    #                 batch_size=64,
    #                 num_epochs=30,
    #                 number_of_gradient_steps=1,
    #                 num_steps_per_epoch=1000,
    #                 max_path_length=max_path_length,
    #                 num_steps_per_eval=300,
    #                 policy_learning_rate=cart_prod[i][0],
    #                 tau=cart_prod[i][1],
    #             ),
    #         },
    #         use_gpu=True,
    #     )
    for i in range(0, 1):
        run_experiment(
            example,
            exp_prefix="ddpg-parallel-sawyer-fixed-end-effector-TEST",
            seed=random.randint(0, 666),
            mode='here',
            variant={
                'version': 'Original',
                'max_path_length': max_path_length,
                'use_gpu': True,
                'algo_params': dict(
                    batch_size=64,
                    num_epochs=10,
                    number_of_gradient_steps=1,
                    num_steps_per_epoch=1000,
                    max_path_length=max_path_length,
                    num_steps_per_eval=300,
                ),
            },
            use_gpu=True,
        )
    #  env.turn_off_robot()

def run():
    run_experiment(
        example,
        exp_prefix="ddpg-parallel-sawyer-fixed-end-effector-10-seeds",
        seed=random.randint(0, 666),
        mode='here',
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
                'safety_force_magnitude': 5,
                'temp': 1,
                'remove_action': False,
                'experiment': experiments[2],
                'reward_magnitude': 10,
            },
            'es_params': {
                'max_sigma': .25,
                'min_sigma': .25,
            },
            'algo_params': dict(
                batch_size=64,
                num_epochs=30,
                number_of_gradient_steps=1,
                num_steps_per_epoch=500,
                max_path_length=max_path_length,
                num_steps_per_eval=500,
            ),
        },
        use_gpu=True,
    )