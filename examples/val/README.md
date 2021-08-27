## Visuomotor Affordance Learning (VAL)

Paper: http://arxiv.org/abs/2106.00671

Website: https://sites.google.com/view/val-rl

### Repos
- bullet-manipulation (`roboverse`): https://github.com/anair13/bullet-manipulation-affordances
- bullet-objects: https://github.com/avisingh599/roboverse/tree/master/roboverse/assets/bullet-objects

### Branches
- bullet-manipulation: rig_affordances
- bullet-objects: sasha-master

### Scripts
- examples/val/full1.py
- examples/val/pretrained1.py (for top drawer, bottom drawer [button])
- examples/val/pretrained_objects1.py (for tray, pnp)

### Tests
- tests/regression/val
These minified tests that run quickly check whether the algorithm returns the same numbers on the same data.

### Datasets
Premade data (~50GB) is available at: https://drive.google.com/drive/u/1/folders/1Kq77B8CWEpY3HQHv3FoRjHycS9SjZIw7

A zip file (2.5GB) available at https://drive.google.com/file/d/1haNopjb0-Qic40YJARnSYiyMGH84j2KK/view?usp=sharing

You can download the zip file on command line with:

`gdown https://drive.google.com/uc?id=1haNopjb0-Qic40YJARnSYiyMGH84j2KK`

#### Scripts to generate data
Object (tray, pnp) Experiment:
- Dataset Collector: https://github.com/anair13/bullet-manipulation-affordances/tree/bullet-metalearning/shapenet_scripts/3dof_gr_demo_collector.py
- Goal Collector: https://github.com/anair13/bullet-manipulation-affordances/tree/bullet-metalearning/shapenet_scripts/rig_presample_goals.py

Drawer + Button Experiment:
- Dataset Collector: https://github.com/anair13/bullet-manipulation-affordances/tree/bullet-metalearning/shapenet_scripts/3dof_afford_demo_collector.py
- Goal Collector: https://github.com/anair13/bullet-manipulation-affordances/tree/bullet-metalearning/shapenet_scripts/rig_afford_presample_goals.py

Tray Experiment:
- Dataset Collector: https://github.com/anair13/bullet-manipulation-affordances/tree/bullet-metalearning/shapenet_scripts/3dof_tray_demo_collector.py
- Goal Collector: https://github.com/anair13/bullet-manipulation-affordances/tree/bullet-metalearning/shapenet_scripts/rig_tray_presample_goals.py

### Baselines:
- VQVAE: experiments/sasha/awac_exps/goal_reaching/combined.py
- CCVAE: experiments/sasha/awac_exps/affordance_baseline/ccvae.py
- VAE: experiments/sasha/awac_exps/affordance_baseline/vae.py
