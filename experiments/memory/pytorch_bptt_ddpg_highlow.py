"""
Try the PyTorch version of BPTT DDPG on HighLow env.
"""
from railrl.envs.memory.continuous_memory_augmented import \
    ContinuousMemoryAugmented
from railrl.envs.memory.high_low import HighLow
from railrl.launchers.launcher_util import (
    run_experiment,
    set_seed,
)


def experiment(variant):
    from railrl.algos.bptt_ddpg_pytorch import BDP
    from railrl.launchers.launcher_util import (
        set_seed,
    )
    seed = variant['seed']
    algo_params = variant['algo_params']
    set_seed(seed)
    env = HighLow(num_steps=2)
    env = ContinuousMemoryAugmented(
        env,
        num_memory_states=20,
    )
    algorithm = BDP(
        env,
        **algo_params
    )
    algorithm.train()


if __name__ == '__main__':
    n_seeds = 1
    mode = "here"
    exp_prefix = "dev-pytorch"

    algo_params = dict(
        subtraj_length=2,
    )
    variant = dict(
        algo_params=algo_params,
    )
    exp_id = -1
    for seed in range(n_seeds):
        exp_id += 1
        set_seed(seed)
        variant['seed'] = seed
        variant['exp_id'] = exp_id

        run_experiment(
            experiment,
            exp_prefix=exp_prefix,
            seed=seed,
            mode=mode,
            variant=variant,
            exp_id=exp_id,
        )
