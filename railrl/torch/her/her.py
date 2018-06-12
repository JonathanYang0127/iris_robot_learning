import numpy as np
import torch
from railrl.data_management.path_builder import PathBuilder
from railrl.torch.torch_rl_algorithm import TorchRLAlgorithm


class HER(TorchRLAlgorithm):
    """
    Note: this assumes the env will sample the goal when reset() is called,
    i.e. use a "silent" env.

    Hindsight Experience Replay

    This is a template class that should be the first sub-class, i.e.[

    ```
    class HerDdpg(HER, DDPG):
    ```

    and not

    ```
    class HerDdpg(DDPG, HER):
    ```

    Or if you really want to make DDPG the first subclass, do alternatively:
    ```
    class HerDdpg(DDPG, HER):
        def get_batch(self):
            return HER.get_batch(self)
    ```
    for each function defined below.
    """

    def __init__(
            self,
            observation_key=None,
            desired_goal_key=None,
    ):
        self.observation_key = observation_key
        self.desired_goal_key = desired_goal_key

    def _start_new_rollout(self, terminal=True, previous_rollout_last_ob=None):
        self.exploration_policy.reset()
        # Note: we assume we're using a silent env.
        o = self.training_env.reset()
        self._rollout_goal = self.training_env.get_goal()
        return o

    def _handle_step(
            self,
            observation,
            action,
            reward,
            next_observation,
            terminal,
            agent_info,
            env_info,
    ):
        self._current_path_builder.add_all(
            observations=observation,
            actions=action,
            rewards=reward,
            next_observations=next_observation,
            terminals=terminal,
            agent_infos=agent_info,
            env_infos=env_info,
            goals=self._rollout_goal,
        )

    def get_batch(self):
        batch = super().get_batch()
        obs = batch['observations']
        next_obs = batch['next_observations']
        goals = batch['resampled_goals']
        batch['observations'] = torch.cat((
            obs,
            goals
        ), dim=1)
        batch['next_observations'] = torch.cat((
            next_obs,
            goals
        ), dim=1)
        return batch

    def _handle_rollout_ending(self):
        self._n_rollouts_total += 1
        if len(self._current_path_builder) > 0:
            path = self._current_path_builder.get_all_stacked()
            self.replay_buffer.add_path(path)
            self._exploration_paths.append(path)
            self._current_path_builder = PathBuilder()

    def _get_action_and_info(self, observation):
        """
        Get an action to take in the environment.
        :param observation:
        :return:
        """
        self.exploration_policy.set_num_steps_total(self._n_env_steps_total)
        goal = self._rollout_goal
        if self.observation_key:
            observation = observation[self.observation_key]
        if self.desired_goal_key:
            goal = self._rollout_goal[self.desired_goal_key]
        new_obs = np.hstack((observation, goal))
        return self.exploration_policy.get_action(new_obs)

    def get_eval_paths(self):
        paths = []
        n_steps_total = 0
        while n_steps_total <= self.num_steps_per_eval:
            path = self.eval_multitask_rollout()
            paths.append(path)
            n_steps_total += len(path['observations'])
        return paths

    def get_eval_action(self, observation, goal):
        if self.observation_key:
            observation = observation[self.observation_key]
        if self.desired_goal_key:
            goal = goal[self.desired_goal_key]
        new_obs = np.hstack((observation, goal))
        return self.policy.get_action(new_obs)

    def eval_multitask_rollout(self):
        observations = []
        actions = []
        rewards = []
        terminals = []
        agent_infos = []
        env_infos = []
        next_observations = []
        path_length = 0
        # goal = self.env.sample_goal_for_rollout()
        # self.env.set_goal(goal)
        o = self.env.reset()
        if self.render_during_eval:
            self.env.render()
        goal = self.env.get_goal()
        while path_length < self.max_path_length:
            a, agent_info = self.get_eval_action(o, goal)
            next_o, r, d, env_info = self.env.step(a)
            if self.render_during_eval:
                self.env.render()
            observations.append(o)
            rewards.append(r)
            terminals.append(d)
            actions.append(a)
            next_observations.append(next_o)
            agent_infos.append(agent_info)
            env_infos.append(env_info)
            path_length += 1
            if d:
                break
            o = next_o

        actions = np.array(actions)
        if len(actions.shape) == 1:
            actions = np.expand_dims(actions, 1)
        observations = np.array(observations)
        next_observations = np.array(next_observations)
        return dict(
            observations=observations,
            actions=actions,
            rewards=np.array(rewards).reshape(-1, 1),
            next_observations=next_observations,
            terminals=np.array(terminals).reshape(-1, 1),
            agent_infos=agent_infos,
            env_infos=env_infos,
            # goals=np.repeat(goal[None], path_length, 0),
        )
