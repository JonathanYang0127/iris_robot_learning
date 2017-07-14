from pathlib import Path

from railrl.algos.qlearning.state_distance_q_learning import (
    MultitaskPathSampler)
from railrl.data_management.env_replay_buffer import EnvReplayBuffer
from railrl.data_management.split_buffer import SplitReplayBuffer
from railrl.envs.multitask.reacher_env import MultitaskReacherEnv
from railrl.envs.multitask.reacher_simple_state import SimpleReacherEnv
from railrl.envs.wrappers import convert_gym_space
from railrl.exploration_strategies.gaussian_strategy import GaussianStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.policies.zero_policy import ZeroPolicy
from rllab.config import LOG_DIR


def main(variant):
    # env = MultitaskReacherEnv()
    env = SimpleReacherEnv()
    action_space = convert_gym_space(env.action_space)
    # es = OUStrategy(action_space=action_space)
    es = GaussianStrategy(
        action_space=action_space,
        max_sigma=0.2,
        min_sigma=0.2,
    )
    exploration_policy = ZeroPolicy(
        int(action_space.flat_dim),
    )
    pool_size = variant['pool_size']
    pool = SplitReplayBuffer(
        EnvReplayBuffer(
            pool_size,
            env,
            flatten=True,
        ),
        EnvReplayBuffer(
            pool_size,
            env,
            flatten=True,
        ),
        fraction_paths_in_train=0.8,
    )
    sampler = MultitaskPathSampler(
        env,
        exploration_strategy=es,
        exploration_policy=exploration_policy,
        pool=pool,
        **variant['algo_params']
    )
    sampler.collect_data()
    sampler.save_pool()


if __name__ == '__main__':
    out_dir = Path(LOG_DIR) / 'datasets/generated'
    out_dir /= '7-13-simple-reacher-gaussian'
    min_num_steps_to_collect = 100000
    max_path_length = 1000
    pool_size = min_num_steps_to_collect + max_path_length

    # noinspection PyTypeChecker
    variant = dict(
        out_dir=str(out_dir),
        algo_params=dict(
            min_num_steps_to_collect=min_num_steps_to_collect,
            max_path_length=max_path_length,
            render=False,
        ),
        pool_size=pool_size,
    )
    # main(variant)
    run_experiment(
        main,
        exp_prefix='gaussian',
        seed=0,
        mode='here',
        variant=variant,
        exp_id=0,
        use_gpu=True,
        snapshot_mode='last',
        base_log_dir=str(out_dir),
    )
