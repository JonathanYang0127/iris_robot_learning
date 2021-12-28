class EmbeddingWrapperOffline(gym.Env, Serializable):

    def __init__(self, env, embeddings):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.embeddings = embeddings

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        obs.update({'task_embedding': self.embeddings[self.env.task_idx]})
        return obs, rew, done, info

    def reset(self):
        obs = self.env.reset()
        obs.update({'task_embedding': self.embeddings[self.env.task_idx]})
        return obs

    def reset_task(self, task_idx):
        self.env.reset_task(task_idx)

class EmbeddingWrapper(gym.Env, Serializable):
    def __init__(self, env, embeddings):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.embeddings = embeddings
        self.latent_dim = len(self.embeddings[0])
        self.num_tasks = env.num_tasks

    def is_reset_task(self):
        return self.env.is_reset_task()

    def get_task_embedding(self, task_idx):
        if task_idx >= 2 * self.num_tasks:
            return [0] * self.latent_dim
        else:
            return self.embeddings[task_idx]

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        obs.update({'task_embedding': self.get_task_embedding(self.env.task_idx)})
        return obs, rew, done, info

    def reset(self):
        obs = self.env.reset()
        obs.update({'task_embedding': self.get_task_embedding(self.env.task_idx)})
        return obs

    def reset_task(self, task_idx):
        self.env.reset_task(task_idx)
    def get_observation(self):
        return self.env.get_observation()
    def reset_robot_only(self):
        return self.env.reset_robot_only()

    def get_new_task_idx(self):
        '''
        Propose a new task based on whether the last trajectory succeeded.
        '''
        info = self.env.get_info()
        if info['reset_success_target']:
            new_task_idx = self.env.task_idx - self.env.num_tasks
        elif info['place_success_target']:
            new_task_idx = self.env.task_idx + self.env.num_tasks
        else:
            new_task_idx = self.env.task_idx
        return new_task_idx