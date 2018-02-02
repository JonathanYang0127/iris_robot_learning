import argparse
import joblib

from railrl.samplers.util import rollout
from railrl.torch.mpc.collocation.collocation_mpc_controller import SlsqpCMC, \
    GradientCMC
from railrl.torch.mpc.collocation.model_to_implicit_model import \
    ModelToImplicitModel

# 2D point mass
# PATH = '/home/vitchyr/git/railrl/data/local/01-30-dev-mpc-neural-networks/01-30-dev-mpc-neural-networks_2018_01_30_11_28_28_0000--s-24549/params.pkl'
# Reacher 7dof
PATH = '/home/vitchyr/git/railrl/data/local/01-27-reacher-full-mpcnn-H1/01-27-reacher-full-mpcnn-H1_2018_01_27_17_59_04_0000--s-96642/params.pkl'


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
    model = data['model']

    implicit_model = ModelToImplicitModel(
        model,
        # bias=-2
    )
    solver_params = {
        'ftol': 1e-3,
        'maxiter': 100,
    }
    # policy = SlsqpCMC(
    #     implicit_model,
    #     env,
    #     solver_params=solver_params,
    #     planning_horizon=2,
    # )
    policy = GradientCMC(
        implicit_model,
        env,
        planning_horizon=1,
        lagrange_multiplier=10,
        num_particles=100,
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
