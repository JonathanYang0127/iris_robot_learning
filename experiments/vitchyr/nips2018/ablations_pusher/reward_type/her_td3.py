from railrl.envs.mujoco.sawyer_gripper_env import SawyerXYEnv
from railrl.envs.mujoco.sawyer_push_and_reach_env import \
    SawyerPushAndReachXYEasyEnv
from railrl.envs.mujoco.sawyer_push_env import SawyerPushXYEnv
from railrl.envs.multitask.point2d import MultitaskImagePoint2DEnv
from railrl.envs.multitask.pusher2d import FullPusher2DEnv
from railrl.images.camera import sawyer_init_camera

from railrl.launchers.arglauncher import run_variants
import railrl.misc.hyperparameter as hyp
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.vae.relabeled_vae_experiment import experiment

if __name__ == "__main__":
    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev'

    n_seeds = 5
    mode = 'ec2'
    exp_prefix = 'pusher-ablation-reward-type-max-path-length-100'

    vae_paths = {
        "16": "nips-2018-pusher/SawyerXY_vae_for_pushing.pkl",
    }

    variant = dict(
        algo_kwargs=dict(
            num_epochs=505,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            tau=1e-2,
            batch_size=128,
            max_path_length=100,
            discount=0.99,
            min_num_steps_before_training=128,
            # qf_learning_rate=1e-3,
            # policy_learning_rate=1e-4,
        ),
        env=SawyerPushAndReachXYEasyEnv,
        env_kwargs=dict(
            hide_goal=True,
            # reward_info=dict(
            #     type="shaped",
            # ),
        ),
        replay_kwargs=dict(
            fraction_goals_are_rollout_goals=0.2,
            fraction_goals_are_env_goals=0.5,
        ),
        algorithm='Ours',
        normalize=False,
        rdim=32,
        render=False,
        use_env_goals=True,
        vae_paths=vae_paths,
        wrap_mujoco_env=True,
        do_state_based_exp=False,
        exploration_noise=0.1,
        init_camera=sawyer_init_camera,
        version='normal',
        reward_params=dict(
            min_variance=0,
        ),
    )

    search_space = {
        'exploration_type': [
            'ou',
        ],
        'algo_kwargs.num_updates_per_env_step': [4],
        'replay_kwargs.fraction_resampled_goals_are_env_goals': [0.5],
        'replay_kwargs.fraction_goals_are_rollout_goals': [0.2],
        'exploration_noise': [0.2],
        'algo_kwargs.reward_scale': [1e-4],
        'training_mode': ['train'],
        'testing_mode': ['test', ],
        'rdim': [16],
        'reward_params.type': ['latent_distance'],
        'reward_params.min_variance': [0],
        'vae_wrapped_env_kwargs.sample_from_true_prior': [False],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        if (
                variant['replay_kwargs']['fraction_goals_are_rollout_goals'] == 1.0
                and variant['replay_kwargs']['fraction_resampled_goals_are_env_goals'] == 0.5
        ):
            # redundant setting
            continue
        for _ in range(n_seeds):
            run_experiment(
                experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
            )