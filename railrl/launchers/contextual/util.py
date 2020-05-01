from os import path as osp

from railrl.core import logger
from railrl.visualization.video import dump_video


def get_save_video_function(
        rollout_function,
        env,
        policy,
        save_video_period=10,
        imsize=48,
        tag="",
        **dump_video_kwargs
):
    logdir = logger.get_snapshot_dir()

    def save_video(algo, epoch):
        if epoch % save_video_period == 0 or epoch >= algo.num_epochs - 1:
            filename = osp.join(
                logdir,
                'video_{}_{epoch}_env.mp4'.format(tag, epoch=epoch),
            )
            dump_video(env, policy, filename, rollout_function,
                       imsize=imsize, **dump_video_kwargs)
    return save_video


def get_gym_env(env_id, env_class=None, env_kwargs=None):
    if env_kwargs is None:
        env_kwargs = {}

    assert env_id or env_class
    if env_id:
        import gym
        import multiworld
        multiworld.register_all_envs()
        env = gym.make(env_id)
    else:
        env = env_class(**env_kwargs)
    return env