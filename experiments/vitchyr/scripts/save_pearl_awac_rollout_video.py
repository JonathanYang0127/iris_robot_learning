import argparse
import pickle

from rlkit.core import logger
from rlkit.envs.remote import RemoteRolloutEnv
from rlkit.envs.vae_wrappers import VAEWrappedEnv
from rlkit.samplers.rollout_functions import (
    multitask_rollout,
    contextual_rollout,
)
from rlkit.torch.core import PyTorchModule
import rlkit.torch.pytorch_util as ptu
import torch


def simulate_policy(args):
    if args.pause:
        import ipdb; ipdb.set_trace()
    snapshot_path = args.file
    ptu.set_gpu_mode(True)
    data = torch.load(snapshot_path, map_location=ptu.device)
    policy = data['exploration/policy']
    qf1 = data['trainer/qf1']
    qf2 = data['trainer/qf2']
    target_qf1 = data['trainer/target_qf1']
    target_qf2 = data['trainer/target_qf2']
    trainer_policy = data['trainer/policy']
    env = data['evaluation/eval/env']
    observation_key = data['evaluation/eval/observation_key']
    context_keys = data['evaluation/eval/context_keys_for_policy']

    if isinstance(env, RemoteRolloutEnv):
        env = env._wrapped_env
    print("Policy loaded")
    if args.gpu:
        ptu.set_gpu_mode(True)
        policy.to(ptu.device)
    else:
        ptu.set_gpu_mode(False)
        policy.to(ptu.device)
    if isinstance(env, VAEWrappedEnv):
        env.mode(args.mode)
    if args.enable_render or hasattr(env, 'enable_render'):
        # some environments need to be reconfigured for visualization
        env.enable_render()
    if args.multitaskpause:
        env.pause_on_goal = True
    if isinstance(policy, PyTorchModule):
        policy.train(False)
    paths = []
    def check(net):
        for name, param in net.named_parameters():
            if torch.isnan(param).any():
                print(name)
    qf = data['trainer/qf1']
    # import ipdb; ipdb.set_trace()
    observation_key = data.get('evaluation/observation_key', 'observation')
    context_keys = data.get('evaluation/context_keys_for_policy', ['context'])
    context_keys = data.get('evaluation/hard_init/context_keys_for_policy')

    while True:
        paths.append(contextual_rollout(
            env,
            policy,
            max_path_length=args.H,
            render=not args.hide,
            observation_key=observation_key,
            context_keys_for_policy=context_keys,
            # context_keys_for_policy=['state_desired_goal'],
        ))
        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics(paths)
        if hasattr(env, "get_diagnostics"):
            for k, v in env.get_diagnostics(paths).items():
                logger.record_tabular(k, v)
        logger.dump_tabular()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=300,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=10,
                        help='Speedup')
    parser.add_argument('--mode', default='video_env', type=str,
                        help='env mode')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--pause', action='store_true')
    parser.add_argument('--enable_render', action='store_true')
    parser.add_argument('--multitaskpause', action='store_true')
    parser.add_argument('--hide', action='store_true')
    args = parser.parse_args()

    simulate_policy(args)
