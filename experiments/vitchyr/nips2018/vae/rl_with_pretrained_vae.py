import railrl.misc.hyperparameter as hyp
import railrl.torch.pytorch_util as ptu
from railrl.envs.multitask.multitask_env import MultitaskToFlatEnv
from railrl.envs.multitask.point2d import MultitaskImagePoint2DEnv
from railrl.envs.mujoco.sawyer_gripper_env import SawyerXYEnv
from railrl.envs.vae_wrappers import VAEWrappedImageGoalEnv, VAEWrappedEnv
from railrl.envs.wrappers import ImageMujocoEnv
from railrl.envs.wrappers import NormalizedBoxEnv
from railrl.exploration_strategies.base import (
    PolicyWrappedWithExplorationStrategy
)
from railrl.exploration_strategies.epsilon_greedy import EpsilonGreedy
from railrl.exploration_strategies.gaussian_strategy import GaussianStrategy
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.networks import FlattenMlp, TanhMlpPolicy
from railrl.torch.td3.td3 import TD3


def experiment(variant):
    from path import Path
    import joblib
    vae_dir = Path(variant['vae_dir'])
    vae_path = str(vae_dir / 'params.pkl')
    extra_data_path = str(vae_dir / 'extra_data.pkl')
    env = joblib.load(extra_data_path)['env']
    use_env_goals = variant["use_env_goals"]
    render = variant["render"]

    if use_env_goals:
        track_qpos_goal = variant.get("track_qpos_goal", 0)
        env = VAEWrappedImageGoalEnv(env, vae_path, use_vae_obs=True,
                                     use_vae_reward=True, use_vae_goals=True,
                                     render_goals=render, render_rollouts=render, track_qpos_goal=track_qpos_goal)
    else:
        env = VAEWrappedEnv(env, vae_path, use_vae_obs=True,
                            use_vae_reward=True, use_vae_goals=True,
                            render_goals=render, render_rollouts=render)

    env = MultitaskToFlatEnv(env)
    if variant['normalize']:
        env = NormalizedBoxEnv(env)
    exploration_type = variant['exploration_type']
    if exploration_type == 'ou':
        es = OUStrategy(action_space=env.action_space)
    elif exploration_type == 'gaussian':
        es = GaussianStrategy(
            action_space=env.action_space,
            max_sigma=0.1,
            min_sigma=0.1,  # Constant sigma
        )
    elif exploration_type == 'epsilon':
        es = EpsilonGreedy(
            action_space=env.action_space,
            prob_random_action=0.1,
        )
    else:
        raise Exception("Invalid type: " + exploration_type)
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[400, 300],
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[400, 300],
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        hidden_sizes=[400, 300],
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    algorithm = TD3(
        env,
        training_env=env,
        qf1=qf1,
        qf2=qf2,
        policy=policy,
        exploration_policy=exploration_policy,
        **variant['algo_kwargs']
    )
    if ptu.gpu_enabled():
        print("cuda")
        algorithm.cuda()
        env._wrapped_env.vae.cuda()
    else:
        env._wrapped_env.vae.cpu()
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    vae_paths = {
        # "2": "/home/vitchyr/git/railrl/data/local/04-24-dev/04-24-dev_2018_04_24_22_14_04_0000--s-99859/params.pkl",
        # "4": "/home/vitchyr/git/railrl/data/local/04-24-dev/04-24-dev_2018_04_24_22_19_02_0000--s-9523/params.pkl",
        # "16": "/home/vitchyr/git/railrl/data/local/04-24-dev/04-24-dev_2018_04_24_22_28_58_0000--s-52846/params.pkl"
        "8": "/home/vitchyr/git/railrl/data/doodads3/04-25-sawyer-reach-xy-vae-train-2/04-25-sawyer-reach-xy-vae-train-2-id0-s9267/params.pkl",
    }

    variant = dict(
        algo_kwargs=dict(
            num_epochs=100,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            tau=1e-2,
            batch_size=128,
            max_path_length=100,
            discount=0.99,
            # qf_learning_rate=1e-3,
            # policy_learning_rate=1e-4,
        ),
        env_kwargs=dict(
            # render_onscreen=False,
            # render_size=84,
            # ignore_multitask_goal=True,
            # ball_radius=1,
        ),
        algorithm='TD3',
        normalize=False,
        rdim=4,
        render=False,
        # env=MultitaskImagePoint2DEnv,
        env=SawyerXYEnv,
        use_env_goals=True,
        vae_dir="/home/vitchyr/git/railrl/data/local/04-25-dev-sawyer-reacher-vae-train-2/04-25-dev-sawyer-reacher-vae-train-2_2018_04_25_14_37_09_0000--s-26862/",
    )

    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev-point2d-with-pretrained-vae'

    # n_seeds = 3
    # mode = 'ec2'
    exp_prefix = 'sawyer-reacher-use-env-goal'

    search_space = {
        'exploration_type': [
            'ou',
        ],
        'algo_kwargs.reward_scale': [1e-6],
        'rdim': [8],
        'seedid': range(n_seeds),
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
            )
