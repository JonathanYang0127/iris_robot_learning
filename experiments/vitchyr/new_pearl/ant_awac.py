"""
PEARL Experiment
"""

import click

from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.pearl.new_awac_launcher import pearl_awac_experiment
import rlkit.misc.hyperparameter as hyp


@click.command()
@click.option('--debug', is_flag=True, default=False)
@click.option('--exp_name', default=None)
@click.option('--mode', default='local')
@click.option('--gpu', default=False)
@click.option('--gpu_id', default=0)
@click.option('--nseeds', default=1)
def main(debug, exp_name, mode, gpu, gpu_id, nseeds):
    variant = dict(
        replay_buffer_kwargs=dict(
            max_replay_buffer_size=1000000,
            use_next_obs_in_context=False,
            sparse_rewards=False,
        ),
        name_to_expl_path_collector_kwargs=dict(
            prior=dict(
                accum_context=False,
                resample_latent_period=1,
                update_posterior_period=0,
                use_predicted_reward=False,
            ),
            posterior=dict(
                accum_context=False,
                resample_latent_period=1,
                update_posterior_period=0,
                use_predicted_reward=False,
            ),
        ),
        name_to_eval_path_collector_kwargs=dict(
            prior=dict(
                accum_context=False,
                resample_latent_period=1,
                update_posterior_period=0,
                use_predicted_reward=False,
            ),
            posterior=dict(
                accum_context=False,
                resample_latent_period=1,
                update_posterior_period=0,
                use_predicted_reward=False,
            ),
            posterior_live_update=dict(
                accum_context=False,
                resample_latent_period=1,
                update_posterior_period=1,
                use_predicted_reward=False,
            ),
        ),
        qf_kwargs=dict(
            hidden_sizes=[256, 256],
        ),
        policy_kwargs=dict(
            hidden_sizes=[256, 256, 256, 256],
            max_log_std=0,
            min_log_std=-6,
            std_architecture="values",
        ),
        policy_class='GaussianPolicy',
        pretrain_rl=True,
        pearl_buffer_kwargs=dict(
            meta_batch_size=4,
            embedding_batch_size=256,
        ),
        load_buffer_kwargs={
            "pretrain_buffer_path": "demos/ant_dir/buffer_500k/extra_snapshot_itr100.pkl"
        },
        saved_tasks_path="demos/ant_dir/buffer_500k/tasks.pkl",
        pretrain_offline_algo_kwargs={
            "batch_size": 128,
            "logging_period": 1000,
            "meta_batch_size": 4,
            "num_batches": 50000,
            "task_embedding_batch_size": 64
        },
        env_name='ant-dir',
        n_train_tasks=2,
        n_eval_tasks=2,
        latent_dim=5,  # dimension of the latent context vector
        path_to_weights=None, # path to pre-trained weights to load into networks
        env_params=dict(
            n_tasks=2, # number of distinct tasks in this domain, shoudl equal sum of train and eval tasks
            randomize_tasks=True, # shuffle the tasks after creating them
        ),
        trainer_kwargs=dict(
            soft_target_tau=0.005, # for SAC target network update
            policy_lr=3E-4,
            qf_lr=3E-4,
            context_lr=3e-4,
            kl_lambda=.1, # weight on KL divergence term in encoder loss
            use_information_bottleneck=True, # False makes latent context deterministic
            use_next_obs_in_context=False, # use next obs if it is useful in distinguishing tasks
            sparse_rewards=False, # whether to sparsify rewards as determined in env
            recurrent=False, # recurrent or permutation-invariant encoder
            discount=0.99, # RL discount factor
            reward_scale=5., # scale rewards before constructing Bellman update, effectively controls weight on the entropy of the policy
        ),
        algo_kwargs=dict(
            num_epochs=500, # number of data sampling / training iterates
            num_trains_per_train_loop=4000,
            num_eval_steps_per_epoch=1000,
            num_expl_steps_per_train_loop=1000,
            num_train_loops_per_epoch=1,
            batch_size=1024,  # number of transitions in the RL batch
            max_path_length=1000,  # max path length for this environment
            min_num_steps_before_training=1000,
            # update_post_train=1, # how often to resample the context when collecting data during training (in trajectories)
            # num_exp_traj_eval=2, # how many exploration trajs to collect before beginning posterior sampling at test time
            # dump_eval_paths=False, # whether to save evaluation trajectories
            # num_iterations_with_reward_supervision=999999,
        ),
    )
    if debug:
       variant['algo_kwargs'].update(dict(
           num_trains_per_train_loop=20,
           num_eval_steps_per_epoch=4*2,
           num_expl_steps_per_train_loop=10,
           max_path_length=2,
           batch_size=13,
           num_epochs=2,
           min_num_steps_before_training=20,
       ))

    mode = mode or 'local'
    exp_name = 'new-pearl--' + (exp_name or 'dev')

    search_space = {
        '_debug_do_not_sqrt': [
            False,
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(nseeds):
            variant['exp_id'] = exp_id
            run_experiment(
                pearl_awac_experiment,
                unpack_variant=True,
                exp_name=exp_name,
                mode=mode,
                variant=variant,
                time_in_mins=3*24*60-1,
                use_gpu=gpu,
                gpu_id=gpu_id,
            )
    print(exp_name)


if __name__ == "__main__":
    main()

