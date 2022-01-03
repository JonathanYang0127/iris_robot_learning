import rlkit.torch.pytorch_util as ptu
from rlkit.misc.roboverse_utils import add_reward_filtered_data_to_buffers_multitask, dump_video_basic,
    get_buffer_size
from rlkit.exploration_strategies import *
from rlkit.samplers.data_collector import ContextualObsDictPathCollector
from rlkit.torch.sac.awac_trainer import AWACTrainer


from rlkit.core.roboverse_serializable import Serializable
import roboverse
import torch
import numpy as np
import argparse
import os
import gym


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint-path", type=str, required=True)
    parser.add_argument("--env", type=str)
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--num-tasks", type=int)
    parser.add_argument("--tsteps", type=int, default=20)
    parser.add_argument("--num-trajectories", type=int, default=50000)
    parser.add_argument("--action-noise", type=float, default=0.0)
    parser.add_argument("--video-save-frequency", type=int, default=20)
    parser.add_argument("--buffer", type=str, required=True)
    parser.add_argument('-e', '--exploration-strategy', type=str,
        choices=('gaussian', 'gaussian_filtered', 'cem', 'fast'))
    parser.add_argument('-u', '--update_frequency', type=int, default=20)
    parser.add_argument("--use-robot-state", default=False, action='store_true')
    args = parser.parse_args()

    expl_env = roboverse.make(args.env, transpose_image=True)
    ptu.set_gpu_mode(True)

    observation_keys = ['image', 'task_embedding']
    if args.use_robot_state:
        observation_keys.append('state')
        state_observation_dim = expl_env.observation_space.spaces['state'].low.size
    else:
        state_observation_dim = 0

    _, ext = os.path.splitext(args.checkpoint_path)
    with open(args.checkpoint_path, 'rb') as handle:
        if ext == ".pt":
            params = torch.load(handle)
            eval_policy = params['evaluation/policy']
            q_function = params['trainer/qf1']
        elif ext == ".pkl":
            eval_policy = pickle.load(handle)
    eval_policy.eval()

    num_tasks = args.num_tasks

    with open(args.buffer, 'rb') as fl:
        data = np.load(fl, allow_pickle=True)
    num_transitions = get_buffer_size(data)
    max_replay_buffer_size = num_transitions + 10
    num_traj_total = len(data)
    latent_dim = data[0]['observations'][0]['task_embedding'].shape[0]
    task_embeddings = dict()
    for i in range(num_tasks):
        task_embeddings[i] = []
    for j in range(num_traj_total):
        task_idx = data[j]['env_infos'][0]['task_idx'] 
        task_embeddings[task_idx].append(data[j]['observations'][0]['task_embeddings'])
    for i in range(variant['num_tasks'])
        task_embeddings[i] = np.asarray(task_embeddings[i])
        task_embeddings[i] = np.mean(task_embeddings[i], axis=0)
    eval_env.observation_space.spaces.update(
        {'task_embedding': spaces.Box(
            low=np.array([-100] * latent_dim),
            high=np.array([100] * latent_dim)
        )})
    eval_env = EmbeddingWrapper(eval_env, embeddings=task_embeddings)
    expl_env = eval_env

    replay_buffer = ObsDictMultiTaskReplayBuffer(
        max_replay_buffer_size,
        expl_env,
        np.arange(num_tasks),
        use_next_obs_in_context=False,
        sparse_rewards = False,
        observation_keys=observation_keys
    )

    add_multitask_data_to_multitask_buffer_v2(data, replay_buffer, observation_keys, num_tasks)
    trainer = AWACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        buffer_policy=buffer_policy,
        multitask=True,
        **variant['trainer_kwargs']
    )

    embeddings = []
    for i in range(len(data)):
        for j in range(len(data[i]['observations'])):
            embeddings.append(data[i]['observations'][j]['task_embedding'])
    embeddings = np.array(embeddings)

    if args.exploration_strategy == 'gaussian':
        exploration_strategy = GaussianExplorationStrategy(embeddings, policy=eval_policy,
            q_function=q_function, n_components=10)
    elif args.exploration_strategy == 'gaussian_filtered':
        exploration_strategy = GaussianExplorationStrategy(embeddings, policy=eval_policy,
            q_function=q_function, n_components=10)
    elif args.exploration_strategy == 'cem':
        exploration_strategy = CEMExplorationStrategy(embeddings, update_frequency=args.update_frequency,
            n_components=10)
    elif args.exploration_strategy == 'fast':
        exploration_strategy = FastExplorationStrategy(embeddings, update_frequency=args.update_frequency,
            n_components=10)
    else:
        raise NotImplementedError

    expl_path_collector = ContextualObsDictPathCollector(
        expl_env,
        eval_policy,
        observation_keys=observation_keys,
    )

    paths = []
    for j in range(args.num_trajectories):
        embedding_kwargs = {'reverse': False}
        embedding = exploration_strategy.sample_embedding(**embedding_kwargs)
        #print(embedding)
        new_paths = expl_path_collector.collect_new_paths(args.tsteps, args.tsteps,
            True, context=embedding, multi_task=True, task_index=args.task)
        post_trajectory_kwargs = {'reverse': False, 
            'embedding': embedding,
            'success': np.sum(new_paths[0]['rewards']) > 0}
        print(post_trajectory_kwargs)
        exploration_strategy.post_trajectory_update(plot=True, **post_trajectory_kwargs)

        paths.append(new_paths)
        if args.video_save_frequency != 0 and j % args.video_save_frequency == 0:
            dump_video_basic(f'videos/{j}', new_paths)

