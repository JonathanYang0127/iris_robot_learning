import numpy as np

from rlkit.torch.pearl.agent import MakePEARLAgentDeterministic
from rlkit.torch.sac.policies import MakeDeterministic
import rlkit.torch.pytorch_util as ptu



class PEARLInPlacePathSampler(object):
    """
    A sampler that does not serialization for sampling. Instead, it just uses
    the current policy and environment as-is.

    WARNING: This will affect the environment! So
    ```
    sampler = InPlacePathSampler(env, ...)
    sampler.obtain_samples  # this has side-effects: env will change!
    ```
    """
    def __init__(self, env, policy, max_path_length):
        self.env = env
        self.policy = policy

        self.max_path_length = max_path_length

    def start_worker(self):
        pass

    def shutdown_worker(self):
        pass

    def obtain_samples(self, deterministic=False, max_samples=np.inf, max_trajs=np.inf, accum_context=True, resample=1, use_predicted_reward=False, task_idx=0):
        """
        Obtains samples in the environment until either we reach either max_samples transitions or
        num_traj trajectories.
        The resample argument specifies how often (in trajectories) the agent will resample it's context.
        """
        assert max_samples < np.inf or max_trajs < np.inf, "either max_samples or max_trajs must be finite"
        policy = MakePEARLAgentDeterministic(self.policy) if deterministic else self.policy
        paths = []
        n_steps_total = 0
        n_trajs = 0
        while n_steps_total < max_samples and n_trajs < max_trajs:
            path = rollout(
                self.env, policy,
                task_idx=task_idx,
                max_path_length=self.max_path_length, accum_context=accum_context, use_predicted_reward=use_predicted_reward)
            # save the latent context that generated this trajectory
            # path['context'] = policy.z.detach().cpu().numpy()
            paths.append(path)
            n_steps_total += len(path['observations'])
            n_trajs += 1
            # don't we also want the option to resample z ever transition?
            # if n_trajs % resample == 0:
            #     policy.sample_z()
        return paths, n_steps_total


def rollout(
        env,
        agent,
        task_idx,
        max_path_length=np.inf,
        accum_context=True,
        animated=False,
        save_frames=False,
        use_predicted_reward=False,
        resample_latent_period=1,
    ):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos

    :param env:
    :param agent:
    :task_idx: the task index
    :param max_path_length:
    :param accum_context: if True, accumulate the collected context
    :param animated:
    :param save_frames: if True, save video of rollout
    :param resample_latent_period: How often to resample the latent posterior.
        If zero, never resample.
    :return:
    """
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    contexts = []
    env.reset_task(task_idx)
    o = env.reset()
    next_o = None

    if animated:
        env.render()
    context = []
    z = agent.latent_prior.sample()
    for path_length in range(max_path_length):
        a, agent_info = agent.get_action(o, z)
        next_o, r, d, env_info = env.step(a)
        if use_predicted_reward:
            r = agent.infer_reward(o, a, z)
        if accum_context:
            context.append([o, a, r, next_o, d, env_info])
        if resample_latent_period and path_length % resample_latent_period == 0 and len(context) > 0:
            latent_posterior = agent.latent_posterior(context)
            z = latent_posterior.rsample()
        contexts.append(ptu.get_numpy(z))
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        o = next_o
        if animated:
            env.render()
        if save_frames:
            from PIL import Image
            image = Image.fromarray(np.flipud(env.get_image()))
            env_info['frame'] = image
        env_infos.append(env_info)
        if d:
            break

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
        cotnexts=np.array(contexts),
    )


def split_paths(paths):
    """
    Stack multiples obs/actions/etc. from different paths
    :param paths: List of paths, where one path is something returned from
    the rollout functino above.
    :return: Tuple. Every element will have shape batch_size X DIM, including
    the rewards and terminal flags.
    """
    rewards = [path["rewards"].reshape(-1, 1) for path in paths]
    terminals = [path["terminals"].reshape(-1, 1) for path in paths]
    actions = [path["actions"] for path in paths]
    obs = [path["observations"] for path in paths]
    next_obs = [path["next_observations"] for path in paths]
    rewards = np.vstack(rewards)
    terminals = np.vstack(terminals)
    obs = np.vstack(obs)
    actions = np.vstack(actions)
    next_obs = np.vstack(next_obs)
    assert len(rewards.shape) == 2
    assert len(terminals.shape) == 2
    assert len(obs.shape) == 2
    assert len(actions.shape) == 2
    assert len(next_obs.shape) == 2
    return rewards, terminals, obs, actions, next_obs


def split_paths_to_dict(paths):
    rewards, terminals, obs, actions, next_obs = split_paths(paths)
    return dict(
        rewards=rewards,
        terminals=terminals,
        observations=obs,
        actions=actions,
        next_observations=next_obs,
    )


def get_stat_in_paths(paths, dict_name, scalar_name):
    if len(paths) == 0:
        return np.array([[]])

    if type(paths[0][dict_name]) == dict:
        # Support rllab interface
        return [path[dict_name][scalar_name] for path in paths]

    return [
        [info[scalar_name] for info in path[dict_name]]
        for path in paths
    ]
