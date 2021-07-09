"""
AWAC PEARL Experiment
"""

import click
import json
import os
import sys

from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.pearl import configs
import rlkit.pythonplusplus as ppp
from rlkit.torch.pearl.awac_launcher import pearl_awac_launcher_simple
import rlkit.util.hyperparameter as hyp


@click.command()
@click.option('--config', default='experiments/references/pearl/cheetah-dir-offline-start.json')
@click.option('--debug', is_flag=True, default=False)
@click.option('--exp_name', default=None)
@click.option('--mode', default='htp')
@click.option('--gpu', default=False)
@click.option('--nseeds', default=1)
def main(config, debug, exp_name, mode, gpu, nseeds):
    if config:
        with open(os.path.join(config)) as f:
            exp_params = json.load(f)
    else:  # You can also hard-code it if you don't give a config file.
        exp_params = {}
    if debug:
        exp_params['algo_params'] = {
            "meta_batch": 4,
            "num_initial_steps": 20,
            "num_steps_prior": 10,
            "num_steps_posterior": 0,
            "num_extra_rl_steps_posterior": 10,
            "num_evals": 4,
            "num_iterations": 20,
            "num_steps_per_eval": 6,
            "num_exp_traj_eval": 2,
            "embedding_batch_size": 256,
            "num_iterations_with_reward_supervision": 10,
            "freeze_encoder_buffer_in_unsupervised_phase": True,
            # "train_reward_pred_in_unsupervised_phase": False,
            "embedding_mini_batch_size": 256,
            "num_train_steps_per_itr": 20,
            "max_path_length": 2,
            "save_replay_buffer": True,
        }
        exp_params['pretrain_offline_algo_kwargs'] = {
            "batch_size": 128,
            "logging_period": 5,
            "meta_batch_size": 2,
            "num_batches": 50,
            "task_embedding_batch_size": 3
        }
        exp_name = 'dev'
        mode = 'local'
        # exp_params["net_size"] = 3
    variant = ppp.merge_recursive_dicts(
        exp_params,
        configs.default_awac_trainer_config,
        ignore_duplicate_keys_in_second_dict=True,
    )

    # s = "experiments/"
    # n = len(s)
    # exp_name = exp_name or sys.argv[0][n:-3]
    exp_name = 'pearl-awac-hc--' + __file__.split('/')[-1].split('.')[0].replace('_', '-')

    search_space = {
        'algo_params.save_replay_buffer': [
            True,
        ],
        'pretrain_rl': [
            True,
            # False,
        ],
        'latent_size': [
            1,
            2,
            5,
            8,
        ],
        'networks_ignore_context': [
            False,
        ],
        'algo_params.num_iterations_with_reward_supervision': [
            9999,
        ],
        'trainer_kwargs.beta': [
            0.5,
            2,
            5,
            10,
            50,
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        variant['algo_kwargs'] = variant.pop('algo_params')
        variant['latent_dim'] = variant.pop('latent_size')
        # net_size = variant.pop('net_size')
        # variant['qf_kwargs'] = dict(
        #     hidden_sizes=[net_size, net_size, net_size],
        # )
        # variant['policy_kwargs'] = dict(
        #     hidden_sizes=[net_size, net_size, net_size],
        # )
        for _ in range(nseeds):
            variant['exp_id'] = exp_id
            run_experiment(
                pearl_awac_launcher_simple,
                unpack_variant=True,
                exp_name=exp_name,
                mode=mode,
                variant=variant,
                time_in_mins=int(2.8 * 24 * 60),  # if you use mode=sss
                use_gpu=gpu,
            )
    print(exp_name)


if __name__ == "__main__":
    main()

