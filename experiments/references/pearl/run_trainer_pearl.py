"""
PEARL Experiment
"""

import click
import json
import os

from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.pearl import configs
import rlkit.pythonplusplus as ppp
from rlkit.torch.pearl.sac_launcher import pearl_experiment
import rlkit.misc.hyperparameter as hyp


@click.command()
@click.argument('config', default=None)
@click.option('--debug', is_flag=True, default=False)
@click.option('--exp_name', default=None)
@click.option('--mode', default='local')
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
        }
        exp_params["net_size"] = 3
    variant = ppp.merge_recursive_dicts(
        exp_params,
        configs.default_trainer_config,
        ignore_duplicate_keys_in_second_dict=True,
    )

    mode = mode or 'local'
    exp_name = exp_name or 'dev'

    search_space = {
        'algo_params.save_replay_buffer': [
            True,
        ],
        # 'algo_params.save_extra_manual_epoch_list': [
        #     [15, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        # ],
        # 'algo_params.num_iterations': [
        #     # 10,
        #     # 20,
        #     50,
        #     500,
        # ],
        'algo_params.num_iterations_with_reward_supervision': [
            # 10,
            # 20,
            # 30,
            9999,
        ],
        # 'algo_params.freeze_encoder_buffer_in_unsupervised_phase': [
        #     True,
        #     # False,
        # ],
        # 'algo_params.train_reward_pred_in_unsupervised_phase': [
        #     # True,
        #     False,
        # ],
        # 'algo_params.use_encoder_snapshot_for_reward_pred_in_unsupervised_phase': [
        #     True,
        #     # False,
        # ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        variant['algo_kwargs'] = variant.pop('algo_params')
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
                pearl_experiment,
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

