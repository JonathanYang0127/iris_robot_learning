"""
Try the PyTorch version of BPTT DDPG on HighLow env.
"""
import random

from railrl.envs.memory.continuous_memory_augmented import (
    ContinuousMemoryAugmented
)
from railrl.envs.memory.high_low import HighLow
from railrl.envs.pygame.water_maze import (
    WaterMaze,
    WaterMazeEasy,
    WaterMazeMemory,
)
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import (
    run_experiment,
)
from railrl.misc.hyperparameter import DeterministicHyperparameterSweeper
from railrl.policies.torch import MemoryPolicy, RWACell
from railrl.pythonplusplus import identity
from railrl.qfunctions.torch import MemoryQFunction
from railrl.torch.bnlstm import LSTMCell, BNLSTMCell


def experiment(variant):
    from railrl.torch.bptt_ddpg import BpttDdpg
    from railrl.launchers.launcher_util import (
        set_seed,
    )
    from railrl.exploration_strategies.product_strategy import ProductStrategy
    seed = variant['seed']
    algo_params = variant['algo_params']
    memory_dim = variant['memory_dim']
    env_class = variant['env_class']
    env_params = variant['env_params']
    memory_aug_params = variant['memory_aug_params']
    qf_params = variant['qf_params']
    policy_params = variant['policy_params']

    es_params = variant['es_params']
    env_es_class = es_params['env_es_class']
    env_es_params = es_params['env_es_params']
    memory_es_class = es_params['memory_es_class']
    memory_es_params = es_params['memory_es_params']

    set_seed(seed)
    raw_env = env_class(**env_params)
    env = ContinuousMemoryAugmented(
        raw_env,
        num_memory_states=memory_dim,
        **memory_aug_params
    )
    env_strategy = env_es_class(
        action_space=raw_env.action_space,
        **env_es_params
    )
    write_strategy = memory_es_class(
        action_space=env.memory_state_space,
        **memory_es_params
    )
    es = ProductStrategy([env_strategy, write_strategy])
    qf = MemoryQFunction(
        int(raw_env.observation_space.flat_dim),
        int(raw_env.action_space.flat_dim),
        memory_dim,
        400,
        300,
        **qf_params,
    )
    policy = MemoryPolicy(
        int(raw_env.observation_space.flat_dim),
        int(raw_env.action_space.flat_dim),
        memory_dim,
        400,
        300,
        **policy_params,
    )
    algorithm = BpttDdpg(
        env,
        es,
        qf=qf,
        policy=policy,
        **algo_params
    )
    algorithm.train()


if __name__ == '__main__':
    n_seeds = 1
    mode = "here"
    exp_prefix = "dev-6-14-pytorch-2"
    run_mode = 'none'

    n_seeds = 10
    mode = "ec2"
    exp_prefix = "paper-6-14-hl-our-method-sweep-subtraj-length-batchsize1000"

    run_mode = 'grid'
    use_gpu = True
    if mode != "here":
        use_gpu = False

    H = 25
    subtraj_length = 25
    num_steps_per_iteration = 100
    num_steps_per_eval = 1000
    num_iterations = 30
    batch_size = 1000
    memory_dim = 30
    version = exp_prefix
    # version = "H = {0}, subtraj length = {1}".format(H, subtraj_length)
    version = "Our Method"
    # noinspection PyTypeChecker
    variant = dict(
        memory_dim=memory_dim,
        # env_class=WaterMazeEasy,
        # env_class=WaterMaze,
        # env_class=WaterMazeMemory,
        env_class=HighLow,
        env_params=dict(
            horizon=H,
        ),
        memory_aug_params=dict(
            max_magnitude=1,
        ),
        algo_params=dict(
            subtraj_length=subtraj_length,
            batch_size=batch_size,
            num_epochs=num_iterations,
            num_steps_per_epoch=num_steps_per_iteration,
            num_steps_per_eval=num_steps_per_eval,
            discount=1.,
            use_gpu=use_gpu,
            action_policy_optimize_bellman=False,
            write_policy_optimizes='both',
            action_policy_learning_rate=1e-3,
            write_policy_learning_rate=1e-5,
            qf_learning_rate=1e-3,
            max_path_length=H,
            refresh_entire_buffer_period=None,
            save_new_memories_back_to_replay_buffer=True,
        ),
        qf_params=dict(
            output_activation=identity,
        ),
        policy_params=dict(
            cell_class=LSTMCell,
        ),
        es_params=dict(
            env_es_class=OUStrategy,
            env_es_params=dict(
                max_sigma=1,
                min_sigma=None,
            ),
            memory_es_class=OUStrategy,
            memory_es_params=dict(
                max_sigma=0,
                min_sigma=None,
            ),
        ),
        version=version,
    )
    if run_mode == 'grid':
        search_space = {
            # 'algo_params.qf_learning_rate': [1e-3, 1e-5],
            # 'algo_params.action_policy_learning_rate': [1e-3, 1e-5],
            # 'algo_params.write_policy_learning_rate': [1e-5, 1e-7],
            # 'algo_params.action_policy_optimize_bellman': [True, False],
            # 'algo_params.write_policy_optimizes': ['qf', 'bellman', 'both'],
            # 'algo_params.refresh_entire_buffer_period': [None, 1],
            # 'es_params.memory_es_params.max_sigma': [0, 1],
            # 'policy_params.cell_class': [LSTMCell, BNLSTMCell, RWACell],
            'algo_params.subtraj_length': [1, 5, 10, 15, 20],
            # 'algo_params.bellman_error_loss_weight': [0.1, 1, 10, 100, 1000],
            # 'algo_params.tau': [1, 0.1, 0.01, 0.001],
        }
        sweeper = DeterministicHyperparameterSweeper(search_space,
                                                     default_parameters=variant)
        for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
            for i in range(n_seeds):
                run_experiment(
                    experiment,
                    exp_prefix=exp_prefix,
                    seed=i,
                    mode=mode,
                    variant=variant,
                    exp_id=exp_id,
                )
    if run_mode == 'custom_grid':
        for exp_id, (
            action_policy_optimize_bellman,
            write_policy_optimizes,
            refresh_entire_buffer_period,
        ) in enumerate([
            (True, 'both', 1),
            (False, 'qf', 1),
            (True, 'both', None),
            (False, 'qf', None),
        ]):
            variant['algo_params']['action_policy_optimize_bellman'] = (
                action_policy_optimize_bellman
            )
            variant['algo_params']['write_policy_optimizes'] = (
                write_policy_optimizes
            )
            variant['algo_params']['refresh_entire_buffer_period'] = (
                refresh_entire_buffer_period
            )
            for seed in range(n_seeds):
                run_experiment(
                    experiment,
                    exp_prefix=exp_prefix,
                    seed=seed,
                    mode=mode,
                    variant=variant,
                    exp_id=exp_id,
                )
    else:
        for _ in range(n_seeds):
            seed = random.randint(0, 10000)
            run_experiment(
                experiment,
                exp_prefix=exp_prefix,
                seed=seed,
                mode=mode,
                variant=variant,
                exp_id=0,
                use_gpu=use_gpu,
                sync_s3_log=True,
                sync_s3_pkl=True,
                periodic_sync_interval=120,
            )