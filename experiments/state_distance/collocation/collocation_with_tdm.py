import argparse
import joblib
import numpy as np

from railrl.envs.multitask.multitask_env import MultitaskToFlatEnv, \
    MultitaskEnvToSilentMultitaskEnv
from railrl.samplers.util import rollout
from railrl.state_distance.util import merge_into_flat_obs
from railrl.torch.core import PyTorchModule
from railrl.torch.mpc.collocation.collocation_mpc_controller import SlsqpCMC, \
    GradientCMC
import railrl.torch.pytorch_util as ptu

# Reacher7dof - TDM
PATH = '/home/vitchyr/git/railrl/data/doodads3/01-23-reacher-full-ddpg' \
       '-tdm-mtau-0/01-23-reacher-full-ddpg-tdm-mtau-0-id1-s49343/params.pkl'
# point2d - TDM
PATH = '/home/vitchyr/git/railrl/data/local/02-01-dev-sac-tdm-launch/02-01-dev-sac-tdm-launch_2018_02_01_16_40_53_0000--s-2210/params.pkl'


class TdmToImplicitModel(PyTorchModule):
    def __init__(self, env, qf, vf):
        self.quick_init(locals())
        super().__init__()
        self.env = env
        self.qf = qf
        self.vf = vf

    def forward(self, states, actions, next_states):
        taus = ptu.np_to_var(
            np.zeros((states.shape[0], 1))
        )
        goals = self.env.convert_obs_to_goals(next_states)
        flat_obs = merge_into_flat_obs(states, goals, taus)
        if self.vf is None:
            return self.qf(flat_obs, actions).sum(1)
        else:
            return self.qf(flat_obs, actions) - self.vf(flat_obs)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str,
                        default=PATH,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=100,
                        help='Max length of rollout')
    parser.add_argument('--nrolls', type=int, default=1,
                        help='Number of rollout per eval')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--mtau', type=float,
                        help='Max tau value')
    parser.add_argument('--hide', action='store_true')
    parser.add_argument('--pause', action='store_true')
    parser.add_argument('--justsim', action='store_true')
    parser.add_argument('--npath', type=int, default=100)
    args = parser.parse_args()
    if args.pause:
        import ipdb; ipdb.set_trace()

    data = joblib.load(args.file)
    env = data['env']
    # env = MultitaskToFlatEnv(env.wrapped_env)
    env = MultitaskEnvToSilentMultitaskEnv(env.wrapped_env)
    qf = data['qf']

    implicit_model = TdmToImplicitModel(
        env,
        qf,
        vf=None,
        # bias=-2
    )
    solver_params = {
        'ftol': 1e-2,
        'maxiter': 100,
    }
    policy = SlsqpCMC(
        implicit_model,
        env,
        # use_implicit_model_gradient=True,
        solver_params=solver_params
    )
    policy = GradientCMC(
        implicit_model,
        env,
        planning_horizon=1,
        lagrange_multiplier=10,
        num_grad_steps=10,
        num_particles=128,
        warm_start=False,
    )
    while True:
        paths = [rollout(
            env,
            policy,
            max_path_length=args.H,
            animated=not args.hide,
        )]
        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics(paths)
