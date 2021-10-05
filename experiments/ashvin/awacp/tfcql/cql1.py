import rlkit.util.hyperparameter as hyp
from rlkit.launchers.arglauncher import run_variants


def train_eval(
    env_name,
    exp_name,
    # Training params
    tpu=False,
    use_gpu=False,
    num_gradient_updates=1000000,
    actor_fc_layers=(256, 256),
    critic_joint_fc_layers=(256, 256, 256),
    # Agent params
    batch_size=256,
    bc_steps=0,
    actor_learning_rate=3e-5,
    critic_learning_rate=3e-4,
    alpha_learning_rate=3e-4,
    reward_scale_factor=1.0,
    cql_alpha_learning_rate=3e-4,
    cql_alpha=5.0,
    cql_tau=10.0,
    num_cql_samples=10,
    reward_noise_variance=0.0,
    include_critic_entropy_term=False,
    use_lagrange_cql_alpha=True,
    log_cql_alpha_clipping=None,
    softmax_temperature=1.0,
    # Data params
    reward_shift=0.0,
    action_clipping=None,
    use_trajectories=False,
    data_shuffle_buffer_size_per_record=1,
    data_shuffle_buffer_size=100,
    data_num_shards=1,
    data_block_length=10,
    data_parallel_reads=None,
    data_parallel_calls=10,
    data_prefetch=10,
    data_cycle_length=10,
    # Others
    policy_save_interval=10000,
    eval_interval=10000,
    summary_interval=1000,
    learner_iterations_per_call=1,
    eval_episodes=10,
    debug_summaries=False,
    summarize_grads_and_vars=False,
    seedid=None):
  """Trains and evaluates CQL-SAC."""
  import numpy as np
  import tensorflow as tf

  from tf_agents.agents.cql import cql_sac_agent
  from tf_agents.agents.ddpg import critic_network
  from tf_agents.agents.sac import tanh_normal_projection_network
  from tf_agents.environments import tf_py_environment
  from tf_agents.experimental.examples.cql_sac.kumar20.d4rl_utils import load_d4rl
  from tf_agents.experimental.examples.cql_sac.kumar20.data_utils import create_tf_record_dataset
  from tf_agents.metrics import py_metrics
  from tf_agents.networks import actor_distribution_network
  from tf_agents.policies import py_tf_eager_policy
  from tf_agents.train import actor
  from tf_agents.train import learner
  from tf_agents.train import triggers
  from tf_agents.train.utils import strategy_utils
  from tf_agents.train.utils import train_utils
  from tf_agents.trajectories import trajectory

  from tf_agents.replay_buffers import tf_uniform_replay_buffer

  import os

  from absl import app
  from absl import flags
  from absl import logging

  logging.set_verbosity(logging.INFO)
  logging.info('Training CQL-SAC on: %s', env_name)
  tf.random.set_seed(seedid)
  np.random.seed(seedid)

  # Load environment.
  env = load_d4rl(env_name)
  tf_env = tf_py_environment.TFPyEnvironment(env)
  strategy = strategy_utils.get_strategy(tpu, use_gpu)

  root_dir = os.getenv('TFAGENTS_ROOT_DIR') # export TFAGENTS_ROOT_DIR=/home/ashvin/tmp/cql_sac
  root_dir = "%s/%s/%s/%s/" % (root_dir, exp_name, env_name, str(seedid))
  dataset_path = os.getenv('TFAGENTS_D4RL_DATA') # export TFAGENTS_D4RL_DATA=/home/ashvin/tmp/d4rl_dataset
  if not dataset_path.endswith('.tfrecord'):
    dataset_path = os.path.join(dataset_path, env_name,
                                '%s*.tfrecord' % env_name)
  logging.info('Loading dataset from %s', dataset_path)
  dataset_paths = tf.io.gfile.glob(dataset_path)

  # Create dataset.
  with strategy.scope():
    dataset = create_tf_record_dataset(
        dataset_paths,
        batch_size,
        shuffle_buffer_size_per_record=data_shuffle_buffer_size_per_record,
        shuffle_buffer_size=data_shuffle_buffer_size,
        num_shards=data_num_shards,
        cycle_length=data_cycle_length,
        block_length=data_block_length,
        num_parallel_reads=data_parallel_reads,
        num_parallel_calls=data_parallel_calls,
        num_prefetch=data_prefetch,
        strategy=strategy,
        reward_shift=reward_shift,
        action_clipping=action_clipping,
        use_trajectories=use_trajectories)

  # Create agent.
  time_step_spec = tf_env.time_step_spec()
  observation_spec = time_step_spec.observation
  action_spec = tf_env.action_spec()
  with strategy.scope():
    train_step = train_utils.create_train_step()

    actor_net = actor_distribution_network.ActorDistributionNetwork(
        observation_spec,
        action_spec,
        fc_layer_params=actor_fc_layers,
        continuous_projection_net=tanh_normal_projection_network
        .TanhNormalProjectionNetwork)

    critic_net = critic_network.CriticNetwork(
        (observation_spec, action_spec),
        joint_fc_layer_params=critic_joint_fc_layers,
        kernel_initializer='glorot_uniform',
        last_kernel_initializer='glorot_uniform')

    agent = cql_sac_agent.CqlSacAgent(
        time_step_spec,
        action_spec,
        actor_network=actor_net,
        critic_network=critic_net,
        actor_optimizer=tf.keras.optimizers.Adam(
            learning_rate=actor_learning_rate),
        critic_optimizer=tf.keras.optimizers.Adam(
            learning_rate=critic_learning_rate),
        alpha_optimizer=tf.keras.optimizers.Adam(
            learning_rate=alpha_learning_rate),
        cql_alpha=cql_alpha,
        num_cql_samples=num_cql_samples,
        include_critic_entropy_term=include_critic_entropy_term,
        use_lagrange_cql_alpha=use_lagrange_cql_alpha,
        cql_alpha_learning_rate=cql_alpha_learning_rate,
        target_update_tau=5e-3,
        target_update_period=1,
        random_seed=seedid,
        cql_tau=cql_tau,
        reward_noise_variance=reward_noise_variance,
        num_bc_steps=bc_steps,
        td_errors_loss_fn=tf.math.squared_difference,
        gamma=0.99,
        reward_scale_factor=reward_scale_factor,
        gradient_clipping=None,
        log_cql_alpha_clipping=log_cql_alpha_clipping,
        softmax_temperature=softmax_temperature,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=train_step)
    agent.initialize()

  # Create learner.
  saved_model_dir = os.path.join(root_dir, learner.POLICY_SAVED_MODEL_DIR)
  collect_env_step_metric = py_metrics.EnvironmentSteps()
  learning_triggers = [
      triggers.PolicySavedModelTrigger(
          saved_model_dir,
          agent,
          train_step,
          interval=policy_save_interval,
          metadata_metrics={
              triggers.ENV_STEP_METADATA_KEY: collect_env_step_metric
          }),
      triggers.StepPerSecondLogTrigger(train_step, interval=100)
  ]
  cql_learner = learner.Learner(
      root_dir,
      train_step,
      agent,
      experience_dataset_fn=lambda: dataset,
      triggers=learning_triggers,
      summary_interval=summary_interval,
      strategy=strategy)

  # Create actor for evaluation.
  eval_greedy_policy = py_tf_eager_policy.PyTFEagerPolicy(
      agent.policy, use_tf_function=True)
  eval_actor = actor.Actor(
      env,
      eval_greedy_policy,
      train_step,
      metrics=actor.eval_metrics(eval_episodes),
      summary_dir=os.path.join(root_dir, 'eval'),
      episodes_per_run=eval_episodes)

  # Run.
  dummy_trajectory = trajectory.mid((), (), (), 0., 1.)
  num_learner_iterations = int(num_gradient_updates /
                               learner_iterations_per_call)
  for _ in range(num_learner_iterations):
    # Mimic collecting environment steps since we loaded a static dataset.
    for _ in range(learner_iterations_per_call):
      collect_env_step_metric(dummy_trajectory)

    cql_learner.run(iterations=learner_iterations_per_call)
    if eval_interval and train_step.numpy() % eval_interval == 0:
      eval_actor.run_and_log()



def main():
  variant = dict(
    env_name='antmaze-medium-play-v0',
    exp_name="cqloffline2",
    learner_iterations_per_call=500,
    policy_save_interval=10000,
    eval_interval=10000,
    summary_interval=1000,
    num_gradient_updates=1000000,
    action_clipping=None,
    use_trajectories=False,
    seedid=0,
    bc_steps = 10000,
    reward_scale_factor = 4.0,
    reward_shift = -0.5,
    critic_learning_rate = 3e-4,
    actor_learning_rate = 1e-4,
    cql_tau = 5.0,
    reward_noise_variance = 0.1,
    log_cql_alpha_clipping = (-1, 100.0),
    launcher_config=dict(
      unpack_variant=True,
    ),
  )

  search_space = {
    'env_name': ["antmaze-umaze-v0", "antmaze-umaze-diverse-v0", "antmaze-medium-play-v0",
        "antmaze-medium-diverse-v0", "antmaze-large-diverse-v0", "antmaze-large-play-v0",
    ],
    'seedid': [None],
  }

  sweeper = hyp.DeterministicHyperparameterSweeper(
      search_space, default_parameters=variant,
  )

  variants = []
  for variant in sweeper.iterate_hyperparameters():
      variants.append(variant)

  run_variants(train_eval, variants, )

if __name__ == "__main__":
  main()
