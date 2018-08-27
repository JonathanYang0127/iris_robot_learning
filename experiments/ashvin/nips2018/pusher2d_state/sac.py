import railrl.misc.hyperparameter as hyp
import railrl.torch.pytorch_util as ptu
from railrl.envs.mujoco.pusher2d import Pusher2DEnv
from railrl.envs.multitask.multitask_env import MultitaskToFlatEnv
from railrl.envs.multitask.pusher2d import CylinderXYPusher2DEnv
from railrl.envs.wrappers import NormalizedBoxEnv
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.networks import FlattenMlp
from railrl.torch.sac.policies import TanhGaussianPolicy
from railrl.torch.sac.sac import SoftActorCritic
from railrl.launchers.arglauncher import run_variants


def experiment(variant):
    if variant['multitask']:
        env = CylinderXYPusher2DEnv(**variant['env_kwargs'])
        env = MultitaskToFlatEnv(env)
    else:
        env = Pusher2DEnv(**variant['env_kwargs'])
    if variant['normalize']:
        env = NormalizedBoxEnv(env)
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size
    qf = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    vf = FlattenMlp(
        input_size=obs_dim,
        output_size=1,
        **variant['vf_kwargs']
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        **variant['policy_kwargs']
    )
    algorithm = SoftActorCritic(
        env=env,
        policy=policy,
        qf=qf,
        vf=vf,
        **variant['algo_kwargs']
    )
    if ptu.gpu_enabled():
        algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_kwargs=dict(
            num_epochs=1000,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            batch_size=128,
            max_path_length=100,
            discount=0.99,
        ),
        env_kwargs=dict(
            randomize_goals=True,
            use_hand_to_obj_reward=False,
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        vf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        algorithm='SAC',
        multitask=True,
        normalize=True,
    )

    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev'

    n_seeds = 3
    mode = 'ec2'
    exp_prefix = 'pusher-2d-state-baselines-h100-multitask-less-shaped'

    search_space = {
        'algo_kwargs.reward_scale': [1, 10, 100],
        'seedid': range(n_seeds),
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    run_variants(experiment, sweeper.iterate_hyperparameters())
