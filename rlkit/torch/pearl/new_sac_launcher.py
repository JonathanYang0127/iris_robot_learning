from collections import OrderedDict
from pathlib import Path

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.data_management.multitask_replay_buffer import MultiTaskReplayBuffer
from rlkit.demos.source.mdp_path_loader import MDPPathLoader
from rlkit.envs.images import GymEnvRenderer
from rlkit.envs.images.plot_renderer import TextRenderer, ScrollingPlotRenderer
from rlkit.envs.pearl_envs import ENVS, register_pearl_envs
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.envs.wrappers.flat_to_dict import FlatToDictEnv
from rlkit.samplers.data_collector.joint_path_collector import \
    JointPathCollector
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.pearl import video
from rlkit.torch.pearl.agent import PEARLAgent, MakePEARLAgentDeterministic
from rlkit.torch.pearl.buffer import PearlReplayBuffer
from rlkit.torch.pearl.diagnostics import (
    DebugInsertImagesEnv,
    FlatToDictPearlPolicy,
)
from rlkit.torch.pearl.diagnostics import get_diagnostics
from rlkit.torch.pearl.networks import MlpEncoder
from rlkit.torch.pearl.path_collector import (
    PearlPathCollector,
    PearlJointPathCollector,
)
from rlkit.torch.pearl.pearl_algorithm import PearlAlgorithm
from rlkit.torch.pearl.pearl_trainer import PEARLSoftActorCriticTrainer
from rlkit.torch.pearl.sampler import rollout
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.visualization.video import dump_video


def pearl_sac_experiment(
        qf_kwargs=None,
        vf_kwargs=None,
        trainer_kwargs=None,
        algo_kwargs=None,
        context_encoder_kwargs=None,
        policy_kwargs=None,
        policy_path=None,
        normalize_env=True,
        env_name=None,
        env_id=None,
        env_class=None,
        env_kwargs=None,
        env_params=None,
        add_env_demos=False,
        path_loader_kwargs=None,
        env_demo_path=None,
        env_offpolicy_data_path=None,
        add_env_offpolicy_data=False,
        exploration_kwargs=None,
        expl_path_collector_kwargs=None,
        pearl_buffer_kwargs=None,
        name_to_eval_path_collector_kwargs=None,
        name_to_expl_path_collector_kwargs=None,
        replay_buffer_class=EnvReplayBuffer,
        replay_buffer_kwargs=None,
        use_validation_buffer=False,
        pretrain_policy=False,
        pretrain_rl=False,
        train_rl=False,
        path_loader_class=MDPPathLoader,
        latent_dim=None,
        # video/debug
        debug=False,
        save_video=False,
        presampled_goals=None,
        renderer_kwargs=None,
        image_env_kwargs=None,
        save_paths=False,
        load_demos=False,
        load_env_dataset_demos=False,
        save_initial_buffers=False,
        save_pretrained_algorithm=False,
        ignore_overlapping_train_and_test=False,
        _debug_do_not_sqrt=False,
        save_video_kwargs=None,
        n_train_tasks_for_video=None,
        n_test_tasks_for_video=None,
        video_img_size=256,
        # PEARL
        n_train_tasks=0,
        n_eval_tasks=0,
        path_to_weights=None,
        util_params=None,
        use_data_collectors=False,
        use_next_obs_in_context=False,
):
    save_video_kwargs = save_video_kwargs or {}
    register_pearl_envs()
    env_kwargs = env_kwargs or {}
    env_params = env_params or {}
    path_loader_kwargs = path_loader_kwargs or {}
    exploration_kwargs = exploration_kwargs or {}
    replay_buffer_kwargs = replay_buffer_kwargs or {}
    context_encoder_kwargs = context_encoder_kwargs or {}
    trainer_kwargs = trainer_kwargs or {}
    if n_train_tasks_for_video is None:
        n_train_tasks_for_video = n_train_tasks
    if n_test_tasks_for_video is None:
        n_test_tasks_for_video = n_eval_tasks
    base_expl_env = ENVS[env_name](**env_params)
    expl_env = NormalizedBoxEnv(base_expl_env)
    base_eval_env = ENVS[env_name](**env_params)
    base_eval_env.tasks = base_expl_env.tasks
    eval_env = NormalizedBoxEnv(base_eval_env)
    reward_dim = 1

    if debug:
        algo_kwargs['max_path_length'] = 50
        algo_kwargs['batch_size'] = 5
        algo_kwargs['num_epochs'] = 5
        algo_kwargs['num_eval_steps_per_epoch'] = 100
        algo_kwargs['num_expl_steps_per_train_loop'] = 100
        algo_kwargs['num_trains_per_train_loop'] = 10
        algo_kwargs['min_num_steps_before_training'] = 100

    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    if use_next_obs_in_context:
        context_encoder_input_dim = 2 * obs_dim + action_dim + reward_dim
    else:
        context_encoder_input_dim = obs_dim + action_dim + reward_dim
    context_encoder_output_dim = latent_dim * 2

    def create_qf():
        return ConcatMlp(
            input_size=obs_dim + action_dim + latent_dim,
            output_size=1,
            **qf_kwargs
        )

    qf1 = create_qf()
    qf2 = create_qf()
    vf = ConcatMlp(
        input_size=obs_dim + latent_dim,
        output_size=1,
        **vf_kwargs
    )

    policy = TanhGaussianPolicy(
        obs_dim=obs_dim + latent_dim,
        action_dim=action_dim,
        **policy_kwargs,
    )
    context_encoder = MlpEncoder(
        input_size=context_encoder_input_dim,
        output_size=context_encoder_output_dim,
        **context_encoder_kwargs
    )
    reward_predictor = ConcatMlp(
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
        hidden_sizes=[200, 200, 200],
    )
    agent = PEARLAgent(
        latent_dim,
        context_encoder,
        policy,
        reward_predictor,
        use_next_obs_in_context=use_next_obs_in_context,
        _debug_do_not_sqrt=_debug_do_not_sqrt,
    )
    trainer = PEARLSoftActorCriticTrainer(
        latent_dim=latent_dim,
        agent=agent,
        qf1=qf1,
        qf2=qf2,
        vf=vf,
        reward_predictor=reward_predictor,
        context_encoder=context_encoder,
        **trainer_kwargs
    )
    task_indices = expl_env.get_all_task_idx()
    train_task_indices = task_indices[:n_train_tasks]
    test_task_indices = task_indices[-n_eval_tasks:]
    if (
            n_train_tasks + n_eval_tasks > len(task_indices)
            and not ignore_overlapping_train_and_test
    ):
        raise ValueError("Your test and train overlap!")
    eval_policy = MakePEARLAgentDeterministic(agent)
    expl_policy = agent

    replay_buffer = MultiTaskReplayBuffer(
        env=expl_env,
        task_indices=task_indices,
        **replay_buffer_kwargs
    )
    enc_replay_buffer = MultiTaskReplayBuffer(
        env=expl_env,
        task_indices=task_indices,
        **replay_buffer_kwargs
    )
    pearl_replay_buffer = PearlReplayBuffer(
        replay_buffer,
        enc_replay_buffer,
        train_task_indices=train_task_indices,
        **pearl_buffer_kwargs
    )

    def create_eval_path_collector(env, policy):
        eval_path_collectors = {
            'train/' + name: PearlPathCollector(
                env, policy, train_task_indices, pearl_replay_buffer, **kwargs)
            for name, kwargs in name_to_eval_path_collector_kwargs.items()
        }
        eval_path_collectors.update({
            'test/' + name: PearlPathCollector(
                env, policy, test_task_indices,
                pearl_replay_buffer,
                **kwargs)
            for name, kwargs in name_to_eval_path_collector_kwargs.items()
        })
        return PearlJointPathCollector(eval_path_collectors)

    eval_path_collector = create_eval_path_collector(eval_env, eval_policy)

    def create_expl_path_collector(env, policy):
        return PearlJointPathCollector({
            name: PearlPathCollector(
                env, policy, train_task_indices,
                pearl_replay_buffer,
                **kwargs)
            for name, kwargs in name_to_expl_path_collector_kwargs.items()
        })
    expl_path_collector = create_expl_path_collector(expl_env, expl_policy)

    diagnostic_fns = get_diagnostics(base_expl_env)
    algorithm = PearlAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=pearl_replay_buffer,
        train_task_indices=train_task_indices,
        test_task_indices=test_task_indices,
        evaluation_get_diagnostic_functions=diagnostic_fns,
        exploration_get_diagnostic_functions=diagnostic_fns,
        **algo_kwargs
    )

    def check_data_collection_settings():
        if (
                algorithm.num_expl_steps_per_train_loop % (
                len(name_to_expl_path_collector_kwargs) * algorithm.max_path_length) != 0
        ):
            raise ValueError("# of exploration steps should divide nicely")
        if (
                algorithm.num_eval_steps_per_epoch % (len(name_to_eval_path_collector_kwargs) * algorithm.max_path_length) != 0
        ):
            raise ValueError("# of eval steps should divide nicely")
    check_data_collection_settings()
    algorithm.to(ptu.device)

    if save_video:
        font_size = int(video_img_size / 256 * 40)  # heuristic
        def config_reward_ax(ax):
            ax.set_title('reward vs step')
            ax.set_xlabel('steps')
            ax.set_ylabel('reward')
            size = font_size
            ax.yaxis.set_tick_params(labelsize=size)
            ax.xaxis.set_tick_params(labelsize=size)
            ax.title.set_size(size)
            ax.xaxis.label.set_size(size)
            ax.yaxis.label.set_size(size)

        def make_video_func(
                env, policy, tag, create_path_collector, num_steps, task_indices
        ):
            obs_key = 'obervation_for_video'
            img_policy = FlatToDictPearlPolicy(policy, obs_key)
            env = FlatToDictEnv(env, obs_key)

            img_renderer = GymEnvRenderer(
                width=video_img_size,
                height=video_img_size,
            )
            text_renderer = TextRenderer(
                text='test',
                width=video_img_size,
                height=video_img_size,
                font_size=font_size,
            )
            reward_plotter = ScrollingPlotRenderer(
                width=video_img_size,
                height=video_img_size,
                modify_ax_fn=config_reward_ax,
            )
            renderers = OrderedDict([
                ('image_observation', img_renderer),
                ('reward_plot', reward_plotter),
                ('text', text_renderer),
            ])
            img_env = DebugInsertImagesEnv(
                wrapped_env=env,
                renderers=renderers,
            )
            video_path_collector = create_path_collector(img_env, img_policy)
            keys_to_save = list(renderers.keys())
            return video.PearlSaveVideoFunction(
                video_path_collector,
                keys_to_save=keys_to_save,
                obs_dict_key='observations',
                image_format=text_renderer.output_image_format,
                text_renderer=text_renderer,
                imsize=video_img_size,
                unnormalize=True,
                task_indices_per_rollout=task_indices,
                tag=tag,
                num_steps=num_steps,
                max_path_length=algorithm.max_path_length,
                **save_video_kwargs
            )
        video_train_tasks = train_task_indices[:n_train_tasks_for_video]
        video_eval_tasks = list(sorted(
            set(test_task_indices[:n_test_tasks_for_video]).union(
                video_train_tasks
            )
        ))  # avoid duplicates
        n_expl_video_rollouts = (
                len(expl_path_collector.path_collectors)
                * len(video_train_tasks)
        )
        save_expl_video_func = make_video_func(
            expl_env,
            eval_policy,
            'expl',
            create_expl_path_collector,
            num_steps=n_expl_video_rollouts * algorithm.max_path_length,
            task_indices=video_train_tasks,
        )
        algorithm.post_train_funcs.append(save_expl_video_func)

        n_eval_video_rollouts = (
                len(eval_path_collector.path_collectors)
                * len(video_eval_tasks)
        )
        save_eval_video_func = make_video_func(
            eval_env,
            eval_policy,
            'eval',
            create_eval_path_collector,
            num_steps=n_eval_video_rollouts * algorithm.max_path_length,
            task_indices=video_eval_tasks,
        )
        algorithm.post_train_funcs.append(save_eval_video_func)
    algorithm.train()
