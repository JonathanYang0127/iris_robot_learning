from visualization.grill.config import (
    output_dir,
    ashvin_base_dir,
    vitchyr_base_dir,
    format_func,
    our_method_name,
)
import matplotlib.pyplot as plt
from railrl.misc import plot_util as plot
from railrl.misc import data_processing as dp

exps = plot.load_exps([ashvin_base_dir + "s3doodad/share/reacher/reacher-abalation-resample-strategy"], suppress_output=True)
# plot.tag_exps(exps, "name", "oracle")
plot.comparison(exps, "Final  distance Mean",
            vary = ["replay_kwargs.fraction_goals_are_env_goals", "replay_kwargs.fraction_goals_are_rollout_goals"],
#           smooth=plot.padded_ma_filter(10),
          ylim=(0.04, 0.2), xlim=(0, 10000), method_order=[2, 1, 0, 3], figsize=(6,5))
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_func))
plt.xlabel("Timesteps")
plt.ylabel("Final Distance to Goal")
plt.title("Visual Reacher")
# plt.legend([]) # ["Ours", "No Relabeling", "HER", "VAE Only", ])
plt.legend([our_method_name, "None", "HER", "VAE Only"],
           bbox_to_anchor=(0.49, -0.2),
           loc="upper center", ncol=4, handlelength=1)
plt.tight_layout()
plt.savefig(output_dir + "reacher_relabeling_ablation.pdf")
print("File saved to", output_dir + "reacher_relabeling_ablation.pdf")
