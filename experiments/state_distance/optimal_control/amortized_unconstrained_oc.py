"""
Amortized version of unconstrained_oc.py
"""
import argparse
import random
import joblib
import os
import numpy as np
from pathlib import Path

from railrl.algos.state_distance.amortized_oc import \
    train_amortized_goal_chooser, AmortizedPolicy
from railrl.envs.multitask.reacher_env import reach_a_point_reward, \
    REACH_A_POINT_GOAL
from railrl.launchers.launcher_util import run_experiment
from railrl.networks.base import Mlp
from railrl.samplers.util import rollout
from rllab.misc import logger
import railrl.torch.pytorch_util as ptu


def experiment(variant):
    num_rollouts = variant['num_rollouts']
    path = variant['qf_path']
    data = joblib.load(path)
    goal_conditioned_model = data['qf']
    env = data['env']
    argmax_qf_policy = data['policy']
    extra_data_path = Path(path).parent / 'extra_data.pkl'
    extra_data = joblib.load(str(extra_data_path))
    replay_buffer = extra_data['replay_buffer']


    """
    Train amortized policy
    """
    goal_chooser = Mlp(
        hidden_sizes=[64, 64],
        output_size=env.goal_dim,
        input_size=int(env.observation_space.flat_dim),
    )
    tau = 5
    if ptu.gpu_enabled():
        goal_chooser.cuda()
        goal_conditioned_model.cuda()
        argmax_qf_policy.cuda()
    train_amortized_goal_chooser(
        goal_chooser,
        goal_conditioned_model,
        argmax_qf_policy,
        env,
        reach_a_point_reward,
        tau,
        replay_buffer,
        learning_rate=1e-3
    )
    policy = AmortizedPolicy(argmax_qf_policy, goal_chooser, tau)

    """
    Eval policy.
    """
    paths = []
    for _ in range(num_rollouts):
        goal = REACH_A_POINT_GOAL
        env.set_goal(goal)
        path = rollout(
            env,
            policy,
            **variant['rollout_params']
        )
        goal_expanded = np.expand_dims(goal, axis=0)
        path['goal_states'] = goal_expanded.repeat(len(path['observations']), 0)
        paths.append(path)
    env.log_diagnostics(paths)
    logger.dump_tabular(with_timestamp=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file with a QF')
    parser.add_argument('--nrolls', type=int, default=5,
                        help='Number of rollouts to do.')
    parser.add_argument('--H', type=int, default=100, help='Horizon.')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--hide', action='store_true')
    parser.add_argument('--discount', type=float, help='Discount Factor')
    parser.add_argument('--nsamples', type=int, default=1000,
                        help='Number of samples for optimization')
    parser.add_argument('--dt', help='decrement tau', action='store_true')
    parser.add_argument('--cycle', help='cycle tau', action='store_true')
    parser.add_argument('--dc', help='decrement and cycle tau',
                        action='store_true')
    args = parser.parse_args()

    n_seeds = 1
    mode = "local"
    exp_prefix = "dev"
    run_mode = 'none'
    use_gpu = True

    discount = 0
    if args.discount is not None:
        print("WARNING: you are overriding the discount factor. Right now "
              "only discount = 0 really makes sense.")
        discount = args.discount

    variant = dict(
        num_rollouts=args.nrolls,
        rollout_params=dict(
            max_path_length=args.H,
            animated=not args.hide,
        ),
        policy_params=dict(
            sample_size=args.nsamples,
        ),
        qf_path=os.path.abspath(args.file),
    )
    if run_mode == 'none':
        for exp_id in range(n_seeds):
            seed = random.randint(0, 999999)
            run_experiment(
                experiment,
                exp_prefix=exp_prefix,
                seed=seed,
                mode=mode,
                variant=variant,
                exp_id=exp_id,
                use_gpu=use_gpu,
            )
