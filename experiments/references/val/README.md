==VAL==

Repos:
railrl-private: https://github.com/vitchyr/railrl-private
bullet-manipulation: https://github.com/JannerM/bullet-manipulation
bullet-objects: https://github.com/Alacarter/bullet-objects

Branches:
railrl-private: val_icra_working
bullet-manipulation: rig_affordances
bullet-objects: sasha-master

Scripts:
experiments/references/full1.py
experiments/references/pretrained1.py
Pretrained VQVAE: https://drive.google.com/file/d/1ohfdGOi6zJc8nxz1Bp9c_XXdhNbH8flS/view?usp=sharing
<details>
  <summary>Older experiments</summary>
Drawer + Button Experiment: https://github.com/vitchyr/railrl-private/blob/val_icra_working/experiments/sasha/awac_exps/goal_reaching/combined.py
Obj Sweep Experiment: https://github.com/vitchyr/railrl-private/blob/val_icra_working/experiments/sasha/awac_exps/goal_reaching/combined_obj_sweep.py
</details>

Datasets:
Premade data is available - ask Ashvin for access.
Object Experiment:
Dataset Collector: https://github.com/JannerM/bullet-manipulation/blob/rig_affordances/shapenet_scripts/3dof_gr_demo_collector.py
Goal Collector: https://github.com/JannerM/bullet-manipulation/blob/rig_affordances/shapenet_scripts/rig_presample_goals.py
Drawer + Button Experiment:
Dataset Collector: https://github.com/JannerM/bullet-manipulation/blob/rig_affordances/shapenet_scripts/3dof_afford_demo_collector.py
Goal Collector: https://github.com/JannerM/bullet-manipulation/blob/rig_affordances/shapenet_scripts/rig_afford_presample_goals.py
Tray Experiment:
Dataset Collector: https://github.com/JannerM/bullet-manipulation/blob/rig_affordances/shapenet_scripts/3dof_tray_demo_collector.py
Goal Collector: https://github.com/JannerM/bullet-manipulation/blob/rig_affordances/shapenet_scripts/rig_tray_presample_goals.py

Baselines:
VQVAE: experiments/sasha/awac_exps/goal_reaching/combined.py
CCVAE: experiments/sasha/awac_exps/affordance_baseline/ccvae.py
VAE: experiments/sasha/awac_exps/affordance_baseline/vae.py
