import gym
import railrl.torch.pytorch_util as ptu
from railrl.data_management.obs_dict_replay_buffer import ObsDictRelabelingBuffer
from railrl.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from railrl.exploration_strategies.gaussian_and_epislon import \
    GaussianAndEpislonStrategy
from railrl.samplers.data_collector import GoalConditionedPathCollector
from railrl.torch.her.her import HERTrainer
from railrl.torch.networks import FlattenMlp, TanhMlpPolicy
from railrl.demos.td3_bc import TD3BCTrainer
from railrl.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from railrl.torch.grill.video_gen import VideoSaveFunction


def state_td3bc_experiment(variant):
    if variant.get('env_id', None):
        env = gym.make(variant['env_id'])
    else:
        env = variant['env_class'](**variant['env_kwargs'])

    expl_env = env # variant['env_class'](**variant['env_kwargs'])
    eval_env = env # variant['env_class'](**variant['env_kwargs'])

    observation_key = 'state_observation'
    desired_goal_key = 'state_desired_goal'
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")
    es = GaussianAndEpislonStrategy(
        action_space=expl_env.action_space,
        max_sigma=.2,
        min_sigma=.2,  # constant sigma
        epsilon=.3,
    )
    obs_dim = expl_env.observation_space.spaces['observation'].low.size
    goal_dim = expl_env.observation_space.spaces['desired_goal'].low.size
    action_dim = expl_env.action_space.low.size
    qf1 = FlattenMlp(
        input_size=obs_dim + goal_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + goal_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + goal_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + goal_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim + goal_dim,
        output_size=action_dim,
        **variant['policy_kwargs']
    )
    target_policy = TanhMlpPolicy(
        input_size=obs_dim + goal_dim,
        output_size=action_dim,
        **variant['policy_kwargs']
    )
    expl_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    replay_buffer = ObsDictRelabelingBuffer(
        env=eval_env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )
    demo_train_buffer = ObsDictRelabelingBuffer(
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )
    demo_test_buffer = ObsDictRelabelingBuffer(
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )
    td3bc_trainer = TD3BCTrainer(
        env=env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        replay_buffer=replay_buffer,
        demo_train_buffer=demo_train_buffer,
        demo_test_buffer=demo_test_buffer,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        target_policy=target_policy,
        **variant['trainer_kwargs']
    )
    trainer = HERTrainer(td3bc_trainer)
    eval_path_collector = GoalConditionedPathCollector(
        eval_env,
        policy,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
    )
    expl_path_collector = GoalConditionedPathCollector(
        expl_env,
        expl_policy,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algo_kwargs']
    )

    if variant.get("save_video", True):
        video_func = VideoSaveFunction(
            env,
            **variant["dump_video_kwargs"],
        )
        algorithm.post_train_funcs.append(video_func)

    algorithm.to(ptu.device)

    td3bc_trainer.load_demos()
    # td3bc_trainer.pretrain_policy_with_bc()
    # td3bc_trainer.pretrain_q_with_bc_data()

    algorithm.train()