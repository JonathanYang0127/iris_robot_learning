from railrl.envs.mujoco.sawyer_gripper_env import SawyerXYEnv
from railrl.envs.mujoco.sawyer_push_env import SawyerPushXYEnv
from railrl.envs.mujoco.sawyer_push_and_reach_env import SawyerPushAndReachXYEasyEnv
from railrl.envs.multitask.point2d import MultitaskImagePoint2DEnv
from railrl.envs.multitask.pusher2d import FullPusher2DEnv
from railrl.images.camera import sawyer_init_camera, sawyer_init_camera_zoomed_in
from railrl.torch.vae.conv_vae import ConvVAE, ConvVAETrainer

from railrl.launchers.arglauncher import run_variants
import railrl.misc.hyperparameter as hyp
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.vae.relabeled_vae_experiment import experiment
import railrl.torch.vae.vae_schedules as vae_schedules

if __name__ == "__main__":

    n_seeds = 1
    mode = 'local'
    exp_prefix = 'pusher-completely-online-stop-after-25'
    rdims = [4]

    vae_paths = {
        # 10 epoch
        #"2": "05-13-sawyer-vae-reacher-recreate-results/05-13-sawyer-vae-reacher-recreate-results_2018_05_13_01_01_38_0000--s-47566-r2/params.pkl",
        #"4": "05-13-sawyer-vae-reacher-recreate-results/05-13-sawyer-vae-reacher-recreate-results_2018_05_13_01_02_03_0000--s-57846-r4/params.pkl",
        #"8": "05-13-sawyer-vae-reacher-recreate-results/05-13-sawyer-vae-reacher-recreate-results_2018_05_13_01_02_27_0000--s-48724-r8/params.pkl",
        #"16": "05-13-sawyer-vae-reacher-recreate-results/05-13-sawyer-vae-reacher-recreate-results_2018_05_13_01_02_53_0000--s-18262-r16/params.pkl",

        # 3 epoch
        #"4": "05-22-online-reacher-three-epoch/05-22-online-reacher-three-epoch_2018_05_22_13_17_07_0000--s-40292-r4/params.pkl",
        # 20 epoch
        #"4": "05-22-online-reacher-twenty-epoch/05-22-online-reacher-twenty-epoch_2018_05_22_17_30_55_0000--s-78057-r4/params.pkl"
        # Completely online
        #"2": "05-13-sawyer-vae-reacher-zero-epoch/05-13-sawyer-vae-reacher-zero-epoch_2018_05_13_03_08_21_0000--s-58139-r2/params.pkl",
        "4": "05-13-sawyer-vae-reacher-zero-epoch/05-13-sawyer-vae-reacher-zero-epoch_2018_05_13_03_08_26_0000--s-54079-r4/params.pkl",
        #"8": "05-13-sawyer-vae-reacher-zero-epoch/05-13-sawyer-vae-reacher-zero-epoch_2018_05_13_03_08_30_0000--s-24696-r8/params.pkl",

        # fully trained
        #"4": "SawyerXY_vae_for_reaching.pkl"

        # Sparse 4 goals
        #"4": "sparse_vae.pkl"

        # fully trained pusher 50 goals
        #"4": "05-29-online-pusher-vae-easy/05-29-online-pusher-vae-easy_2018_05_29_02_31_40_0000--s-14978-r4/params.pkl"
        # fully trained pusher 250 goals
        #"4": "05-29-online-pusher-vae-easy-250/05-29-online-pusher-vae-easy-250_2018_05_29_12_14_34_0000--s-3375-r4/params.pkl"
        # fully trained pusher 2500 goals
        "2500": "05-29-online-pusher-vae-easy-2500/05-29-online-pusher-vae-easy-2500_2018_05_29_12_28_05_0000--s-46268-r4/params.pkl",

        # 16 goals, grid sweep
        #"4": "05-29-pusher-vae-16-sparse-goals/05-29-pusher-vae-16-sparse-goals_2018_05_29_17_29_41_0000--s-84886-r4/params.pkl",
        #"8": "05-29-pusher-vae-16-sparse-goals/05-29-pusher-vae-16-sparse-goals_2018_05_29_22_35_26_0000--s-86061-r8/params.pkl"
        #svae
        #"8": "05-29-pusher-svae-16-sparse-goals/05-29-pusher-svae-16-sparse-goals_2018_05_29_22_45_18_0000--s-2802-r8/params.pkl"


        # pretrained pusher sweep
        "1250": "05-29-vae-for-pusher-1250/05-29-vae-for-pusher-1250_2018_05_29_23_47_35_0000--s-10650-r4/params.pkl",
        #"250": "05-29-vae-for-pusher-250/05-29-vae-for-pusher-250_2018_05_29_23_47_42_0000--s-45315-r4/params.pkl",
        "50": "05-29-vae-for-pusher-50/05-29-vae-for-pusher-50_2018_05_29_23_47_47_0000--s-50577-r4/params.pkl",

        # 96 chosen goals
        "96": "05-30-sparse-vae-for-pusher-24/05-30-sparse-vae-for-pusher-24_2018_05_30_12_33_20_0000--s-69050-r4/params.pkl",
        # denoising vae
            }
    variant = dict(
        algo_kwargs=dict(
            num_epochs=2000,
            min_num_steps_before_training=2000,
            num_steps_per_epoch=200,
            num_steps_per_eval=500,
            tau=1e-2,
            batch_size=128,
            max_path_length=100,
            discount=0.99,
            # qf_learning_rate=1e-3,
            # policy_learning_rate=1e-4,
        ),
        env_kwargs=dict(
            hide_goal=True,
            # reward_info=dict(
            #     type="shaped",
            # ),
        ),
        vae_trainer_kwargs=dict(
            beta=5.0
        ),

        replay_kwargs=dict(
            transformed_goals=True,
            fraction_goals_are_rollout_goals=0.2,
            fraction_goals_are_env_goals=0.5,
            use_image_error=True,
        ),
        algorithm='HER-TD3',
        normalize=False,
        render=False,
        env=SawyerPushAndReachXYEasyEnv,
        use_env_goals=True,
        vae_paths=vae_paths,
        wrap_mujoco_env=True,
        do_state_based_exp=False,
        exploration_noise=0.3,
        init_camera=sawyer_init_camera_zoomed_in,
        online_vae=True,
    )

    search_space = {
        'exploration_type': [
            'ou',
        ],
        'algo_kwargs.num_updates_per_env_step': [1],
        'algo_kwargs.should_train_vae': [vae_schedules.always_train],
        'replay_kwargs.fraction_goals_are_env_goals': [0.5],
        'replay_kwargs.fraction_goals_are_rollout_goals': [0.2],
        'reward_params.type': ['latent_distance'],
        'exploration_noise': [0.4],
        'algo_kwargs.reward_scale': [1e-4],
        'training_mode': ['train'],
        'testing_mode': ['test', ],
        'rdim': rdims,
        'num_online_goals': [250],
        'noise_p': [0.0]
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        if (
                variant['replay_kwargs']['fraction_goals_are_rollout_goals'] == 1.0
                and variant['replay_kwargs']['fraction_goals_are_env_goals'] == 0.5
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
