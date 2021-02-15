# import roboverse
from rlkit.core.simple_offline_rl_algorithm import OfflineMetaRLAlgorithm
from rlkit.data_management.multitask_replay_buffer import MultiTaskReplayBuffer
import rlkit.torch.pytorch_util as ptu
from rlkit.core import logger
from rlkit.core.meta_rl_algorithm import MetaRLAlgorithm
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.demos.source.mdp_path_loader import MDPPathLoader
from rlkit.envs.pearl_envs import ENVS, register_pearl_envs
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.misc.asset_loader import load_local_or_remote_file
from rlkit.samplers.data_collector.joint_path_collector import \
    JointPathCollector
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.pearl.agent import PEARLAgent, MakePEARLAgentDeterministic
from rlkit.torch.pearl.launcher_util import (
    load_buffer_onto_algo,
    policy_class_from_str,
)
from rlkit.torch.pearl.buffer import PearlReplayBuffer
from rlkit.torch.pearl.diagnostics import get_diagnostics
from rlkit.torch.pearl.encoder import MlpEncoder
from rlkit.torch.pearl.path_collector import PearlPathCollector
from rlkit.torch.pearl.pearl_algorithm import PearlAlgorithm
from rlkit.torch.pearl.pearl_awac import PearlAwacTrainer
from rlkit.torch.pearl.pearl_trainer import PEARLSoftActorCriticTrainer
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.torch_rl_algorithm import (
    TorchBatchRLAlgorithm,
)


def pearl_awac_experiment(
        qf_kwargs=None,
        vf_kwargs=None,
        trainer_kwargs=None,
        algo_kwargs=None,
        context_encoder_kwargs=None,
        policy_class="TanhGaussianPolicy",
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
        train_rl=False,
        path_loader_class=MDPPathLoader,
        latent_dim=None,
        # video/debug
        save_video=False,
        presampled_goals=None,
        renderer_kwargs=None,
        image_env_kwargs=None,
        save_paths=False,
        load_demos=False,
        load_env_dataset_demos=False,
        save_initial_buffers=False,
        save_pretrained_algorithm=False,
        _debug_do_not_sqrt=False,
        networks_ignore_context=False,
        use_ground_truth_context=False,
        # PEARL
        n_train_tasks=0,
        n_eval_tasks=0,
        path_to_weights=None,
        util_params=None,
        use_data_collectors=False,
        use_next_obs_in_context=False,
        # Pre-train params
        pretrain_rl=False,
        pretrain_offline_algo_kwargs=None,
        pretrain_buffer_kwargs=None,
        load_buffer_kwargs=None,
        saved_tasks_path=None,
):
    register_pearl_envs()
    env_kwargs = env_kwargs or {}
    env_params = env_params or {}
    path_loader_kwargs = path_loader_kwargs or {}
    exploration_kwargs = exploration_kwargs or {}
    replay_buffer_kwargs = replay_buffer_kwargs or {}
    context_encoder_kwargs = context_encoder_kwargs or {}
    trainer_kwargs = trainer_kwargs or {}
    base_env = ENVS[env_name](**env_params)
    expl_env = NormalizedBoxEnv(base_env)
    eval_env = NormalizedBoxEnv(ENVS[env_name](**env_params))
    reward_dim = 1

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
    target_qf1 = create_qf()
    target_qf2 = create_qf()

    def create_policy():
        if isinstance(policy_class, str):
            final_policy_class = policy_class_from_str(policy_class)
        else:
            final_policy_class = policy_class
        return final_policy_class(
            obs_dim=obs_dim + latent_dim,
            action_dim=action_dim,
            **policy_kwargs)
    policy = create_policy()

    context_encoder = MlpEncoder(
        input_size=context_encoder_input_dim,
        output_size=context_encoder_output_dim,
        hidden_sizes=[200, 200, 200],
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
    eval_policy = MakePEARLAgentDeterministic(agent)
    expl_policy = agent

    if saved_tasks_path:
        task_data = load_local_or_remote_file(
            saved_tasks_path, file_type='joblib')
        tasks = task_data['tasks']
        train_task_indices = task_data['train_task_indices']
        test_task_indices = task_data['eval_task_indices']
        base_env.tasks = tasks
        # task_indices = base_env.get_all_task_idx()
    else:
        tasks = base_env.tasks
        task_indices = base_env.get_all_task_idx()
        train_task_indices = list(task_indices[:n_train_tasks])
        test_task_indices = list(task_indices[-n_eval_tasks:])
        if n_train_tasks + n_eval_tasks > len(task_indices):
            print("WARNING: your test and train overlap!")
    # train_tasks = [tasks[i] for i in train_task_indices]
    # eval_tasks = [tasks[i] for i in test_task_indices]

    trainer = PearlAwacTrainer(
        agent=agent,
        env=expl_env,
        latent_dim=latent_dim,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        reward_predictor=reward_predictor,
        context_encoder=context_encoder,
        _debug_ignore_context=networks_ignore_context,
        _debug_use_ground_truth_context=use_ground_truth_context,
        **trainer_kwargs
    )

    replay_buffer = MultiTaskReplayBuffer(
        env=expl_env,
        task_indices=train_task_indices,
        **replay_buffer_kwargs
    )
    enc_replay_buffer = MultiTaskReplayBuffer(
        env=expl_env,
        task_indices=train_task_indices,
        **replay_buffer_kwargs
    )
    pearl_replay_buffer = PearlReplayBuffer(
        replay_buffer,
        enc_replay_buffer,
        train_task_indices=train_task_indices,
        **pearl_buffer_kwargs
    )

    eval_path_collectors = {
        'train/' + name: PearlPathCollector(
            eval_env, eval_policy, train_task_indices, pearl_replay_buffer, **kwargs)
        for name, kwargs in name_to_eval_path_collector_kwargs.items()
    }
    eval_path_collectors.update({
        'test/' + name: PearlPathCollector(
            eval_env, eval_policy, test_task_indices,
            pearl_replay_buffer,
            **kwargs)
        for name, kwargs in name_to_eval_path_collector_kwargs.items()
    })
    eval_path_collector = JointPathCollector(eval_path_collectors)
    expl_path_collector = JointPathCollector({
        name: PearlPathCollector(
            expl_env, expl_policy, train_task_indices,
            pearl_replay_buffer,
            **kwargs)
        for name, kwargs in name_to_expl_path_collector_kwargs.items()
    })

    diagnostic_fns = get_diagnostics(base_env)
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

    def pretrain_if_needed():
        if load_buffer_kwargs:
            load_buffer_onto_algo(
                replay_buffer,
                enc_replay_buffer,
                **load_buffer_kwargs)

        pretrain_algo = OfflineMetaRLAlgorithm(
            replay_buffer=replay_buffer,
            task_embedding_replay_buffer=enc_replay_buffer,
            trainer=trainer,
            train_tasks=train_task_indices,
            **pretrain_offline_algo_kwargs
        )
        pretrain_algo.to(ptu.device)
        if pretrain_rl:
            logger.remove_tabular_output(
                'progress.csv', relative_to_snapshot_dir=True
            )
            logger.add_tabular_output(
                'pretrain.csv', relative_to_snapshot_dir=True
            )
            pretrain_algo.train()
            logger.remove_tabular_output(
                'pretrain.csv', relative_to_snapshot_dir=True
            )
            logger.add_tabular_output(
                'progress.csv', relative_to_snapshot_dir=True,
            )
    pretrain_if_needed()

    algorithm.to(ptu.device)

    algorithm.train()
