import click
import json

from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.pearl.new_sac_launcher import pearl_sac_experiment
import rlkit.misc.hyperparameter as hyp


@click.command()
@click.option('--debug', is_flag=True, default=False)
@click.option('--exp_name', default=None)
@click.option('--mode', default='local')
@click.option('--gpu', default=False)
@click.option('--gpu_id', default=0)
@click.option('--nseeds', default=1)
def main(debug, exp_name, mode, gpu, gpu_id, nseeds):
    env_config_path = '/home/vitchyr/git/railrl/experiments/vitchyr/new_pearl/configs/point2d.json'
    offline_config_path = '/home/vitchyr/git/railrl/experiments/vitchyr/new_pearl/configs/point2d_offline.json'
    # trainer_config_path = '/home/vitchyr/git/railrl/experiments/vitchyr/new_pearl/configs/trainer.json'
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
        pearl_buffer_kwargs=dict(
            meta_batch_size=4,
            embedding_batch_size=256,
        ),
        env_name='cheetah-dir',
        n_train_tasks=2,
        n_eval_tasks=2,
        latent_size=5, # dimension of the latent context vector
        net_size=300, # number of units per FC layer in each network
        path_to_weights=None, # path to pre-trained weights to load into networks
        env_params=dict(
            n_tasks=2, # number of distinct tasks in this domain, shoudl equal sum of train and eval tasks
            randomize_tasks=True, # shuffle the tasks after creating them
        ),
        trainer_kwargs=dict(
            soft_target_tau=0.005, # for SAC target network update
            policy_lr=3E-4,
            qf_lr=3E-4,
            vf_lr=3E-4,
            context_lr=3e-4,
            kl_lambda=.1, # weight on KL divergence term in encoder loss
            use_information_bottleneck=True, # False makes latent context deterministic
            use_next_obs_in_context=False, # use next obs if it is useful in distinguishing tasks
            sparse_rewards=False, # whether to sparsify rewards as determined in env
            recurrent=False, # recurrent or permutation-invariant encoder
            discount=0.99, # RL discount factor
            reward_scale=5., # scale rewards before constructing Bellman update, effectively controls weight on the entropy of the policy
        ),
    )
    for config_path in [
        env_config_path,
        offline_config_path,
    ]:
        new_variant = json.load(open(config_path, 'rb'))
        variant.update(new_variant)
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
        variant["net_size"] = 3

    exp_name = exp_name or 'new-pearl--' + __file__.split('/')[-1].split('.')[0].replace('_', '-')

    search_space = {
        'algo_kwargs.save_replay_buffer': [
            True,
        ],
        # 'algo_kwargs.save_extra_every_epoch': [
        #     False,
        # ],
        # 'algo_kwargs.save_extra_manual_epoch_list': [
        #     [0, 1, 49, 100, 200, 300, 400, 500],
        # ],
        # 'algo_kwargs.num_iterations': [
        #     # 10,
        #     # 20,
        #     # 30,
        #     # 40,
        #     50,
        #     # 500,
        # ],
        '_debug_do_not_sqrt': [
            False,
        ],
        # 'algo_kwargs.freeze_encoder_buffer_in_unsupervised_phase': [
        #     True,
        #     # False,
        # ],
        # 'algo_kwargs.train_reward_pred_in_unsupervised_phase': [
        #     # True,
        #     False,
        # ],
        # 'algo_kwargs.use_encoder_snapshot_for_reward_pred_in_unsupervised_phase': [
        #     True,
        #     # False,
        # ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        variant['latent_dim'] = variant.pop('latent_size')
        net_size = variant.pop('net_size')
        variant['qf_kwargs'] = dict(
            hidden_sizes=[net_size, net_size, net_size],
        )
        variant['vf_kwargs'] = dict(
            hidden_sizes=[net_size, net_size, net_size],
        )
        variant['policy_kwargs'] = dict(
            hidden_sizes=[net_size, net_size, net_size],
        )
        for _ in range(nseeds):
            variant['exp_id'] = exp_id
            run_experiment(
                pearl_sac_experiment,
                unpack_variant=True,
                exp_name=exp_name,
                mode=mode,
                variant=variant,
                time_in_mins=int(2.8 * 24 * 60),  # if you use mode=sss
                use_gpu=gpu,
                gpu_id=gpu_id,
            )
    print(exp_name)


if __name__ == "__main__":
    main()

