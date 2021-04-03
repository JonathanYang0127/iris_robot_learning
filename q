[1mdiff --git a/experiments/vitchyr/pearl_awac/ant_awac/exp31_sweep_ib_weight.py b/experiments/vitchyr/pearl_awac/ant_awac/exp31_sweep_ib_weight.py[m
[1mindex 2d326686e..f92a5d89f 100644[m
[1m--- a/experiments/vitchyr/pearl_awac/ant_awac/exp31_sweep_ib_weight.py[m
[1m+++ b/experiments/vitchyr/pearl_awac/ant_awac/exp31_sweep_ib_weight.py[m
[36m@@ -61,6 +61,8 @@[m [mdef main(debug, dry, suffix, nseeds):[m
             'trainer_kwargs.kl_lambda': [[m
                 100.,[m
                 10.,[m
[32m+[m[32m                5.,[m
[32m+[m[32m                2.,[m
                 1.,[m
             ],[m
             'trainer_kwargs.backprop_q_loss_into_encoder': [[m
[1mdiff --git a/experiments/vitchyr/pearl_awac/ant_awac/exp32_repeat_exp27_with_resample_latent_period1_update_posterior0.py b/experiments/vitchyr/pearl_awac/ant_awac/exp32_repeat_exp27_with_resample_latent_period1_update_posterior0.py[m
[1mnew file mode 100644[m
[1mindex 000000000..204f9a5d5[m
[1m--- /dev/null[m
[1m+++ b/experiments/vitchyr/pearl_awac/ant_awac/exp32_repeat_exp27_with_resample_latent_period1_update_posterior0.py[m
[36m@@ -0,0 +1,151 @@[m
[32m+[m[32m"""[m
[32m+[m[32mPEARL Experiment[m
[32m+[m[32m"""[m
[32m+[m
[32m+[m[32mimport click[m
[32m+[m[32mfrom pathlib import Path[m
[32m+[m
[32m+[m[32mfrom rlkit.launchers.launcher_util import run_experiment, load_pyhocon_configs[m
[32m+[m[32mimport rlkit.pythonplusplus as ppp[m
[32m+[m[32mfrom rlkit.torch.pearl.cql_launcher import pearl_cql_experiment[m
[32m+[m[32mimport rlkit.misc.hyperparameter as hyp[m
[32m+[m[32mfrom rlkit.torch.pearl.sac_launcher import pearl_sac_experiment[m
[32m+[m[32mfrom rlkit.torch.pearl.awac_launcher import pearl_awac_experiment[m
[32m+[m
[32m+[m
[32m+[m[32mname_to_exp = {[m
[32m+[m[32m    'CQL': pearl_cql_experiment,[m
[32m+[m[32m    'AWAC': pearl_awac_experiment,[m
[32m+[m[32m    'SAC': pearl_sac_experiment,[m
[32m+[m[32m}[m
[32m+[m
[32m+[m
[32m+[m[32m@click.command()[m
[32m+[m[32m@click.option('--debug', is_flag=True, default=False)[m
[32m+[m[32m@click.option('--dry', is_flag=True, default=False)[m
[32m+[m[32m@click.option('--suffix', default=None)[m
[32m+[m[32m@click.option('--nseeds', default=1)[m
[32m+[m[32mdef main(debug, dry, suffix, nseeds):[m
[32m+[m[32m    mode = 'sss'[m
[32m+[m[32m    gpu = True[m
[32m+[m
[32m+[m[32m    base_dir = Path(__file__).parent.parent[m
[32m+[m
[32m+[m[32m    path_parts = __file__.split('/')[m
[32m+[m[32m    suffix = '' if suffix is None else '--{}'.format(suffix)[m
[32m+[m[32m    exp_name = 'pearl-awac-{}--{}{}'.format([m
[32m+[m[32m        path_parts[-2].replace('_', '-'),[m
[32m+[m[32m        path_parts[-1].split('.')[0].replace('_', '-'),[m
[32m+[m[32m        suffix,[m
[32m+[m[32m    )[m
[32m+[m
[32m+[m[32m    if debug or dry:[m
[32m+[m[32m        exp_name = 'dev--' + exp_name[m
[32m+[m[32m        mode = 'local'[m
[32m+[m[32m        nseeds = 1[m
[32m+[m
[32m+[m[32m    print(exp_name)[m
[32m+[m[32m    exp_id = 0[m
[32m+[m
[32m+[m[32m    def run_sweep(search_space, variant, xid):[m
[32m+[m[32m        for k, v in {[m
[32m+[m[32m            'load_buffer_kwargs.start_idx': [[m
[32m+[m[32m                -10000,[m
[32m+[m[32m                -100000,[m
[32m+[m[32m            ],[m
[32m+[m[32m            'load_buffer_kwargs.end_idx': [[m
[32m+[m[32m                200000[m
[32m+[m[32m            ],[m
[32m+[m[32m            'trainer_kwargs.train_context_decoder': [[m
[32m+[m[32m                True,[m
[32m+[m[32m            ],[m
[32m+[m[32m            'trainer_kwargs.backprop_q_loss_into_encoder': [[m
[32m+[m[32m                False,[m
[32m+[m[32m            ],[m
[32m+[m[32m            'algo_kwargs.num_iterations_with_reward_supervision': [[m
[32m+[m[32m                None,[m
[32m+[m[32m            ],[m
[32m+[m[32m            'online_trainer_kwargs.awr_weight': [[m
[32m+[m[32m                1.0,[m
[32m+[m[32m            ],[m
[32m+[m[32m            'online_trainer_kwargs.reparam_weight': [[m
[32m+[m[32m                1.0,[m
[32m+[m[32m            ],[m
[32m+[m[32m            'online_trainer_kwargs.use_reparam_update': [[m
[32m+[m[32m                True,[m
[32m+[m[32m                False,[m
[32m+[m[32m            ],[m
[32m+[m[32m            'online_trainer_kwargs.use_awr_update': [[m
[32m+[m[32m                True,[m
[32m+[m[32m                False,[m
[32m+[m[32m            ],[m
[32m+[m[32m        }.items():[m
[32m+[m[32m            search_space[k] = v[m
[32m+[m[32m        sweeper = hyp.DeterministicHyperparameterSweeper([m
[32m+[m[32m            search_space, default_parameters=variant,[m
[32m+[m[32m        )[m
[32m+[m[32m        for _, variant in enumerate(sweeper.iterate_hyperparameters()):[m
[32m+[m[32m            for _ in range(nseeds):[m
[32m+[m[32m                if ([m
[32m+[m[32m                        not variant['online_trainer_kwargs']['use_awr_update'][m
[32m+[m[32m                        and not variant['online_trainer_kwargs']['use_reparam_update'][m
[32m+[m[32m                ):[m
[32m+[m[32m                    continue[m
[32m+[m[32m                variant['exp_id'] = xid[m
[32m+[m[32m                xid += 1[m
[32m+[m[32m                run_experiment([m
[32m+[m[32m                    name_to_exp[variant['tags']['method']],[m
[32m+[m[32m                    unpack_variant=True,[m
[32m+[m[32m                    exp_name=exp_name,[m
[32m+[m[32m                    mode=mode,[m
[32m+[m[32m                    variant=variant,[m
[32m+[m[32m                    time_in_mins=3 * 24 * 60 - 1,[m
[32m+[m[32m                    use_gpu=gpu,[m
[32m+[m[32m                )[m
[32m+[m[32m        return xid[m
[32m+[m
[32m+[m[32m    def cql_sweep(xid):[m
[32m+[m[32m        configs = [[m
[32m+[m[32m            base_dir / 'configs/default_cql.conf',[m
[32m+[m[32m            base_dir / 'configs/offline_pretraining.conf',[m
[32m+[m[32m            base_dir / 'configs/ant_four_dir_offline.conf',[m
[32m+[m[32m            ][m
[32m+[m[32m        if debug:[m
[32m+[m[32m            configs.append(base_dir / 'configs/debug.conf')[m
[32m+[m[32m        variant = ppp.recursive_to_dict(load_pyhocon_configs(configs))[m
[32m+[m[32m        search_space = {[m
[32m+[m[32m            'trainer_kwargs.with_lagrange': [[m
[32m+[m[32m                True,[m
[32m+[m[32m            ],[m
[32m+[m[32m            'trainer_kwargs.min_q_weight': [[m
[32m+[m[32m                10.0,[m
[32m+[m[32m            ],[m
[32m+[m[32m        }[m
[32m+[m[32m        return run_sweep(search_space, variant, xid)[m
[32m+[m
[32m+[m[32m    def awac_sweep(xid):[m
[32m+[m[32m        configs = [[m
[32m+[m[32m            base_dir / 'configs/default_awac.conf',[m
[32m+[m[32m            base_dir / 'configs/offline_pretraining.conf',[m
[32m+[m[32m            base_dir / 'configs/ant_four_dir_offline.conf',[m
[32m+[m[32m            ][m
[32m+[m[32m        if debug:[m
[32m+[m[32m            configs.append(base_dir / 'configs/debug.conf')[m
[32m+[m[32m        variant = ppp.recursive_to_dict(load_pyhocon_configs(configs))[m
[32m+[m[32m        search_space = {[m
[32m+[m[32m            'trainer_kwargs.beta': [[m
[32m+[m[32m                100,[m
[32m+[m[32m            ],[m
[32m+[m[32m        }[m
[32m+[m[32m        return run_sweep(search_space, variant, xid)[m
[32m+[m
[32m+[m[32m    # exp_id = cql_sweep(exp_id)[m
[32m+[m[32m    exp_id = awac_sweep(exp_id)[m
[32m+[m[32m    print(exp_name, exp_id)[m
[32m+[m
[32m+[m
[32m+[m
[32m+[m
[32m+[m[32mif __name__ == "__main__":[m
[32m+[m[32m    main()[m
[32m+[m
[1mdiff --git a/experiments/vitchyr/pearl_awac/configs/default_base.conf b/experiments/vitchyr/pearl_awac/configs/default_base.conf[m
[1mindex 669b40bba..840a34339 100644[m
[1m--- a/experiments/vitchyr/pearl_awac/configs/default_base.conf[m
[1m+++ b/experiments/vitchyr/pearl_awac/configs/default_base.conf[m
[36m@@ -21,7 +21,7 @@[m
     },[m
     algo_kwargs: {[m
         meta_batch: 16, // number of tasks to average the gradient across[m
[31m-        num_iterations: 500, // number of data sampling / training iterates[m
[32m+[m[32m        num_iterations: 501, // number of data sampling / training iterates[m
         num_initial_steps: 2000, // number of transitions collected per task before training[m
         num_tasks_sample: 5, // number of randomly sampled tasks to collect data for each iteration[m
         num_steps_prior: 400, // number of transitions to collect per task with z ~ prior[m
[36m@@ -38,7 +38,8 @@[m
         num_exp_traj_eval: 1, // how many exploration trajs to collect before beginning posterior sampling at test time[m
         dump_eval_paths: false, // whether to save evaluation trajectories[m
         num_iterations_with_reward_supervision: null,[m
[31m-        save_extra_manual_epoch_list: [0, 1, 49, 100, 200, 300, 400, 500],[m
[32m+[m[32m        save_extra_manual_epoch_list: [0, 50, 100, 200, 300, 400, 500],[m
[32m+[m[32m        save_extra_manual_beginning_epoch_list: [0],[m
         save_replay_buffer: true,[m
         save_algorithm: true,[m
     },[m
[1mdiff --git a/rlkit/core/meta_rl_algorithm.py b/rlkit/core/meta_rl_algorithm.py[m
[1mindex 8d7952bab..10f5983ea 100644[m
[1m--- a/rlkit/core/meta_rl_algorithm.py[m
[1m+++ b/rlkit/core/meta_rl_algorithm.py[m
[36m@@ -58,6 +58,7 @@[m [mclass MetaRLAlgorithm(metaclass=abc.ABCMeta):[m
             num_iterations_with_reward_supervision=np.inf,[m
             freeze_encoder_buffer_in_unsupervised_phase=True,[m
             save_extra_manual_epoch_list=(),[m
[32m+[m[32m            save_extra_manual_beginning_epoch_list=(),[m
             save_extra_every_epoch=False,[m
             use_ground_truth_context=False,[m
             exploration_data_collector=None,[m
[36m@@ -74,6 +75,7 @@[m [mclass MetaRLAlgorithm(metaclass=abc.ABCMeta):[m
         """[m
         self._save_extra_every_epoch = save_extra_every_epoch[m
         self.save_extra_manual_epoch_list = save_extra_manual_epoch_list[m
[32m+[m[32m        self.save_extra_manual_beginning_epoch_list = save_extra_manual_beginning_epoch_list[m
         self.use_encoder_snapshot_for_reward_pred_in_unsupervised_phase = ([m
             use_encoder_snapshot_for_reward_pred_in_unsupervised_phase[m
         )[m
[36m@@ -507,6 +509,12 @@[m [mclass MetaRLAlgorithm(metaclass=abc.ABCMeta):[m
         self._exploration_paths = [][m
         self._do_train_time = 0[m
         logger.push_prefix('Iteration #%d | ' % epoch)[m
[32m+[m[32m        if epoch in self.save_extra_manual_beginning_epoch_list:[m
[32m+[m[32m            logger.save_extra_data([m
[32m+[m[32m                self.get_extra_data_to_save(epoch),[m
[32m+[m[32m                file_name='extra_snapshot_beginning_itr{}'.format(epoch),[m
[32m+[m[32m                mode='cloudpickle',[m
[32m+[m[32m            )[m
 [m
     def _end_epoch(self, epoch):[m
         for post_train_func in self.post_train_funcs:[m
[36m@@ -567,7 +575,7 @@[m [mclass MetaRLAlgorithm(metaclass=abc.ABCMeta):[m
                 accum_context=True,[m
                 initial_context=init_context,[m
                 task_idx=idx,[m
[31m-                resample_latent_period=0,  # following PEARL protocol[m
[32m+[m[32m                resample_latent_period=1,  # following PEARL protocol[m
                 update_posterior_period=0,  # following PEARL protocol[m
                 infer_posterior_at_start=infer_posterior_at_start,[m
             )[m
[1mdiff --git a/rlkit/core/rl_algorithm.py b/rlkit/core/rl_algorithm.py[m
[1mindex 2d9619f1c..d92f40f02 100644[m
[1m--- a/rlkit/core/rl_algorithm.py[m
[1m+++ b/rlkit/core/rl_algorithm.py[m
[36m@@ -39,6 +39,7 @@[m [mclass BaseRLAlgorithm(object, metaclass=abc.ABCMeta):[m
             save_replay_buffer=False,[m
             save_logger=False,[m
             save_extra_manual_epoch_list=(),[m
[32m+[m[32m            save_extra_manual_beginning_epoch_list=(),[m
             keep_only_last_extra=True,[m
     ):[m
         self.trainer = trainer[m
[36m@@ -55,6 +56,7 @@[m [mclass BaseRLAlgorithm(object, metaclass=abc.ABCMeta):[m
         self.save_algorithm = save_algorithm[m
         self.save_replay_buffer = save_replay_buffer[m
         self.save_extra_manual_epoch_list = save_extra_manual_epoch_list[m
[32m+[m[32m        self.save_extra_manual_beginning_epoch_list = save_extra_manual_beginning_epoch_list[m
         self.save_logger = save_logger[m
         self.keep_only_last_extra = keep_only_last_extra[m
         if exploration_get_diagnostic_functions is None:[m
[36m@@ -98,6 +100,8 @@[m [mclass BaseRLAlgorithm(object, metaclass=abc.ABCMeta):[m
 [m
     def _begin_epoch(self):[m
         timer.reset()[m
[32m+[m[32m        if self.epoch in self.save_extra_manual_beginning_epoch_list:[m
[32m+[m[32m            self.save_extra_snapshot()[m
 [m
     def _end_epoch(self):[m
         for post_train_func in self.post_train_funcs:[m
[36m@@ -112,25 +116,28 @@[m [mclass BaseRLAlgorithm(object, metaclass=abc.ABCMeta):[m
             post_epoch_func(self, self.epoch)[m
 [m
         if self.epoch in self.save_extra_manual_epoch_list:[m
[31m-            if self.keep_only_last_extra:[m
[31m-                file_name = 'extra_snapshot'[m
[31m-                info_lines = [[m
[31m-                    'extra_snapshot_itr = {}'.format(self.epoch),[m
[31m-                    'snapshot_dir = {}'.format(logger.get_snapshot_dir())[m
[31m-                ][m
[31m-                logger.save_extra_data([m
[31m-                    '\n'.join(info_lines),[m
[31m-                    file_name='snapshot_info',[m
[31m-                    mode='txt',[m
[31m-                )[m
[31m-            else:[m
[31m-                file_name = 'extra_snapshot_itr{}'.format(self.epoch)[m
[32m+[m[32m            self.save_extra_snapshot()[m
[32m+[m[32m        self.epoch += 1[m
[32m+[m
[32m+[m[32m    def save_extra_snapshot(self, tag=''):[m
[32m+[m[32m        if self.keep_only_last_extra:[m
[32m+[m[32m            file_name = 'extra_snapshot' + tag[m
[32m+[m[32m            info_lines = [[m
[32m+[m[32m                'extra_snapshot_itr = {}'.format(self.epoch),[m
[32m+[m[32m                'snapshot_dir = {}'.format(logger.get_snapshot_dir())[m
[32m+[m[32m            ][m
             logger.save_extra_data([m
[31m-                self.get_extra_data_to_save(self.epoch),[m
[31m-                file_name=file_name,[m
[31m-                mode='cloudpickle',[m
[32m+[m[32m                '\n'.join(info_lines),[m
[32m+[m[32m                file_name='snapshot_info',[m
[32m+[m[32m                mode='txt',[m
             )[m
[31m-        self.epoch += 1[m
[32m+[m[32m        else:[m
[32m+[m[32m            file_name = 'extra_snapshot_itr{}'.format(self.epoch)[m
[32m+[m[32m        logger.save_extra_data([m
[32m+[m[32m            self.get_extra_data_to_save(self.epoch),[m
[32m+[m[32m            file_name=file_name,[m
[32m+[m[32m            mode='cloudpickle',[m
[32m+[m[32m        )[m
 [m
     def _get_snapshot(self):[m
         snapshot = {}[m
[1mdiff --git a/rlkit/envs/pearl_envs/ant_dir.py b/rlkit/envs/pearl_envs/ant_dir.py[m
[1mindex d799fd0eb..78d063c39 100644[m
[1m--- a/rlkit/envs/pearl_envs/ant_dir.py[m
[1m+++ b/rlkit/envs/pearl_envs/ant_dir.py[m
[36m@@ -23,6 +23,7 @@[m [mclass AntDirEnv(MultitaskAntEnv):[m
         super(AntDirEnv, self).__init__(task, n_tasks, **kwargs)[m
 [m
     def step(self, action):[m
[32m+[m[32m        # import ipdb; ipdb.set_trace()[m
         torso_xyz_before = np.array(self.get_body_com("torso"))[m
 [m
         if self.direction_in_degrees:[m
