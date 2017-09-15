import ray

from railrl.envs.remote import RemoteRolloutEnv
from railrl.envs.wrappers import convert_gym_space
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.algos.parallel_naf import ParallelNAF
from railrl.torch.naf import NAF, NafPolicy
from railrl.torch import pytorch_util as ptu
from os.path import exists
import joblib
from railrl.envs.ros.sawyer_env import SawyerEnv
from railrl.torch import pytorch_util as ptu
import random
import itertools

def example(variant):
    env_class = variant['env_class']
    env_params = variant['env_params']
    use_batch_norm=variant['use_batch_norm']
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
    policy_class = variant['policy_class']
    policy_params = dict(
        obs_dim=int(obs_space.flat_dim),
        action_dim=int(action_space.flat_dim),
        hidden_size=100,
        use_batchnorm=use_batch_norm
    )
    naf_policy = NafPolicy(**policy_params)
    remote_env = RemoteRolloutEnv(
            env_class,
            env_params,
            policy_class,
            policy_params,
            es_class,
            es_params,
            variant['max_path_length'],
            variant['normalize_env'],
    )
    algorithm = ParallelNAF(
        remote_env,
        naf_policy=naf_policy,
        exploration_strategy=es,
        **variant['algo_params']
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
learning_rates=[1e-2, 1e-3, 1e-4]
use_batch_norm=[True,False]
cart_prod = list(itertools.product(learning_rates, use_batch_norm))

if __name__ == "__main__":
    ray.init()
    # for i in range(1, len(cart_prod)):
    #     run_experiment(
    #         example,
    #         exp_prefix="naf-parallel-sawyer-fixed-end-effector-hyper-param-search",
    #         seed=random.randint(0, 666),
    #         mode='here',
    #         variant={
    #             'version': 'Original',
    #             'max_path_length': max_path_length,
    #             'use_gpu': True,
    #             'es_class': OUStrategy,
    #             'env_class': SawyerEnv,
    #             'policy_class': NafPolicy,
    #             'normalize_env': False,
    #             'use_batch_norm':cart_prod[i][1],
    #             'env_params': {
    #                 'arm_name': 'right',
    #                 'safety_box': True,
    #                 'loss': 'huber',
    #                 'huber_delta': 10,
    #                 'safety_force_magnitude': 5,
    #                 'temp': 1,
    #                 'remove_action': False,
    #                 'experiment': experiments[2],
    #                 'reward_magnitude': 10,
    #             },
    #             'es_params': {
    #                 'max_sigma': .5,
    #                 'min_sigma': .5,
    #             },
    #             'algo_params': dict(
    #                 batch_size=64,
    #                 num_epochs=30,
    #                 num_steps_per_epoch=1000,
    #                 max_path_length=max_path_length,
    #                 num_steps_per_eval=300,
    #                 naf_policy_learning_rate=cart_prod[i][0],
    #             ),
    #         },
    #         use_gpu=True,
    #     )
    for i in range(0, 3):
        run_experiment(
            example,
            exp_prefix="naf-parallel-sawyer-fixed-end-effector-final",
            seed=random.randint(0, 666),
            mode='here',
            variant={
                'version': 'Original',
                'max_path_length': max_path_length,
                'use_gpu': True,
                'es_class': OUStrategy,
                'env_class': SawyerEnv,
                'policy_class': NafPolicy,
                'normalize_env': False,
                'use_batch_norm':True,
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
                    'max_sigma': .5,
                    'min_sigma': .5,
                },
                'algo_params': dict(
                    batch_size=64,
                    num_epochs=30,
                    num_steps_per_epoch=1000,
                    max_path_length=max_path_length,
                    num_steps_per_eval=300,
                    naf_policy_learning_rate=0.001,
                ),
            },
            use_gpu=True,
        )
