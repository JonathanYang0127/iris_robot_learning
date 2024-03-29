import rlkit.util.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment
import os

from rlkit.util.io import sync_down


def experiment(variant):
    from rlkit.core import logger
    demo_path = sync_down(variant['demo_path'])
    off_policy_path = sync_down(variant['off_policy_path'])
    logdir = logger.get_snapshot_dir()
    os.system('python -m BEAR.main' +
              ' --demo_data='+demo_path+
              ' --off_policy_data='+off_policy_path+
              ' --eval_freq='+variant['eval_freq']+
              ' --algo_name='+variant['algo_name']+
              ' --env_name='+variant['env_name']+
              ' --log_dir='+logdir+
              ' --lagrange_thresh='+variant['lagrange_thresh']+
              ' --distance_type='+variant['distance_type']+
              ' --mode='+variant['mode']+
              ' --num_samples_match='+variant['num_samples_match']+
              ' --lamda='+variant['lambda_']+
              ' --version='+variant['version']+
              ' --mmd_sigma='+variant['mmd_sigma']+
              ' --kernel_type='+variant['kernel_type']+
              ' --use_ensemble_variance='+variant['use_ensemble_variance'])

if __name__ == "__main__":
    variant = dict(
        demo_path='demos/ant_action_noise_15.npy',
        off_policy_path='demos/ant_off_policy_15_demos_100.npy',
        eval_freq='1000',
        algo_name='BEAR',
        env_name='Ant-v2',
        lagrange_thresh='10.0',
        distance_type='MMD',
        mode='auto',
        num_samples_match='5',
        lambda_='0.0',
        version='0.0',
        mmd_sigma='10.0',
        kernel_type='laplacian',
        use_ensemble_variance='"False"',
    )

    search_space = {
        'mmd_sigma':['10.0', '20.0'],
        'num_samples_match':['5', '10', '20'],

    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    n_seeds = 1
    mode = 'local'
    exp_name = 'test'

    # n_seeds = 1
    # mode = 'ec2'
    # exp_name = 'ant_bear_sweep_v1'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                experiment,
                exp_name=exp_name,
                mode=mode,
                unpack_variant=False,
                variant=variant,
                num_exps_per_instance=1,
                use_gpu=False,
                gcp_kwargs=dict(
                    preemptible=False,
                ),
            )
