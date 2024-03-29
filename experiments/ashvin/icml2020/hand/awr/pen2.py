"""
AWR + SAC from demo experiment
"""

from rlkit.demos.source.dict_to_mdp_path_loader import DictToMDPPathLoader
from rlkit.demos.source.mdp_path_loader import MDPPathLoader
from rlkit.launchers.experiments.ashvin.awr_tf import experiment

import rlkit.util.hyperparameter as hyp
from rlkit.launchers.arglauncher import run_variants

from rlkit.torch.sac.policies import GaussianPolicy
from rlkit.util.ml_util import PiecewiseLinearSchedule, ConstantSchedule

if __name__ == "__main__":
    variant = dict(
        launcher_kwargs=dict(
            num_exps_per_instance=1,
            region='us-west-2',
        ),
        max_iter = 1000,
        env='pen-v0',
        agent_configs=dict(
            temp=1.0,
        ),
    )

    search_space = {
        'seedid': range(3),
    }

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)

    run_variants(experiment, variants, run_id=0)
