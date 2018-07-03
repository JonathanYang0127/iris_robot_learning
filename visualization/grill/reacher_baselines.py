from visualization.grill.config import (
    output_dir,
    ashvin_base_dir,
    vitchyr_base_dir,
    format_func
)
import matplotlib.pyplot as plt
from railrl.misc import plot_util as plot

f = plot.filter_by_flat_params({'algo_kwargs.num_updates_per_env_step': 4,
    'replay_kwargs.fraction_goals_are_env_goals': 0.5})

oracle = plot.load_exps([ashvin_base_dir +
    "s3doodad/share/reacher/reacher-baseline-oracle"], suppress_output=True)
plot.tag_exps(oracle, "name", "oracle")

ours = plot.load_exps([ashvin_base_dir +
    "s3doodad/share/reacher/reacher-main-results-ours"], suppress_output=True)
plot.tag_exps(ours, "name", "ours")
f = plot.filter_by_flat_params({'replay_kwargs.fraction_goals_are_env_goals':
    0.0, 'reward_params.type': 'latent_distance'})

her = plot.load_exps([ashvin_base_dir + "s3doodad/share/reward-reaching-sweep"],
        f, suppress_output=True)
plot.tag_exps(her, "name", "her")

dsae = plot.load_exps([ashvin_base_dir+
    's3doodad/share/steven/no-relabeling-test'], suppress_output=True)
plot.tag_exps(dsae, "name", "dsae")
lr = plot.load_exps([vitchyr_base_dir +
    "papers/nips2018/autoencoder_result/05-25-sawyer-reacher-autoencoder-ablation-final/"],
    suppress_output=True)
plot.tag_exps(lr, "name", "l&r")

plot.comparison(ours + oracle + her + lr + dsae, "Final  distance Mean", 
                    vary=["name"],
                    #           smooth=plot.padded_ma_filter(10),
                              method_order=[4, 0, 1, 3, 2], ylim=(0.0, 0.25),
                              xlim=(0, 10000),
)
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_func))
plt.xlabel("Timesteps")
plt.ylabel("Final Distance to Goal")
plt.title("Visual Reacher Baselines")
plt.legend([]) # ["GRiLL", "DSAE", "HER", "Oracle", "L&R", ])

plt.tight_layout()
plt.savefig(output_dir + "reacher_baselines.pdf")
print("File saved to", output_dir + "reacher_baselines.pdf")
