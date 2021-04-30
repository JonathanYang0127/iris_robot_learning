import glob
import re
from collections import OrderedDict
from pathlib import Path
from typing import List

import numpy as np

import rlkit.torch.pytorch_util as ptu
from rlkit.core.meta_rl_algorithm import MetaRLAlgorithm
from rlkit.misc import eval_util
from rlkit.misc.asset_loader import load_local_or_remote_file
from rlkit.torch.sac.policies import GaussianPolicy, TanhGaussianPolicy

ENV_PARAMS = {
    'HalfCheetah-v2': {
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'env_demo_path': dict(
            path="demos/icml2020/mujoco/hc_action_noise_15.npy",
            obs_dict=False,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            path="demos/icml2020/mujoco/hc_off_policy_15_demos_100.npy",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
    },
    'Ant-v2': {
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'env_demo_path': dict(
            path="demos/icml2020/mujoco/ant_action_noise_15.npy",
            obs_dict=False,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            path="demos/icml2020/mujoco/ant_off_policy_15_demos_100.npy",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
    },
    'Walker2d-v2': {
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'env_demo_path': dict(
            path="demos/icml2020/mujoco/walker_action_noise_15.npy",
            obs_dict=False,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            path="demos/icml2020/mujoco/walker_off_policy_15_demos_100.npy",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
    },

    'SawyerRigGrasp-v0': {
        'env_id': 'SawyerRigGrasp-v0',
        # 'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 50,
        # 'num_epochs': 1000,
    },

    'pen-binary-v0': {
        'env_id': 'pen-binary-v0',
        'max_path_length': 200,
        'sparse_reward': True,
        'env_demo_path': dict(
            path="demos/icml2020/hand/pen2_sparse.npy",
            # path="demos/icml2020/hand/sparsity/railrl_pen-binary-v0_demos.npy",
            obs_dict=True,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            # path="demos/icml2020/hand/pen_bc_sparse1.npy",
            # path="demos/icml2020/hand/pen_bc_sparse2.npy",
            # path="demos/icml2020/hand/pen_bc_sparse3.npy",
            # path="demos/icml2020/hand/pen_bc_sparse4.npy",
            path="demos/icml2020/hand/pen_bc_sparse4.npy",
            # path="ashvin/icml2020/hand/sparsity/bc/pen-binary1/run10/id*/video_*_*.p",
            # sync_dir="ashvin/icml2020/hand/sparsity/bc/pen-binary1/run10",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
    },
    'door-binary-v0': {
        'env_id': 'door-binary-v0',
        'max_path_length': 200,
        'sparse_reward': True,
        'env_demo_path': dict(
            path="demos/icml2020/hand/door2_sparse.npy",
            # path="demos/icml2020/hand/sparsity/railrl_door-binary-v0_demos.npy",
            obs_dict=True,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            # path="demos/icml2020/hand/door_bc_sparse1.npy",
            # path="demos/icml2020/hand/door_bc_sparse3.npy",
            path="demos/icml2020/hand/door_bc_sparse4.npy",
            # path="ashvin/icml2020/hand/sparsity/bc/door-binary1/run10/id*/video_*_*.p",
            # sync_dir="ashvin/icml2020/hand/sparsity/bc/door-binary1/run10",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
    },
    'relocate-binary-v0': {
        'env_id': 'relocate-binary-v0',
        'max_path_length': 200,
        'sparse_reward': True,
        'env_demo_path': dict(
            path="demos/icml2020/hand/relocate2_sparse.npy",
            # path="demos/icml2020/hand/sparsity/railrl_relocate-binary-v0_demos.npy",
            obs_dict=True,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            # path="demos/icml2020/hand/relocate_bc_sparse1.npy",
            path="demos/icml2020/hand/relocate_bc_sparse4.npy",
            # path="ashvin/icml2020/hand/sparsity/bc/relocate-binary1/run10/id*/video_*_*.p",
            # sync_dir="ashvin/icml2020/hand/sparsity/bc/relocate-binary1/run10",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
    },
}


def policy_class_from_str(policy_class):
    if policy_class == 'GaussianPolicy':
        return GaussianPolicy
    elif policy_class == 'TanhGaussianPolicy':
        return TanhGaussianPolicy
    else:
        raise ValueError(policy_class)


def load_macaw_buffer_onto_algo(
        algo: MetaRLAlgorithm,
        base_directory: str,
        train_task_idxs: List[int],
        start_idx=0,
        end_idx=None,
):
    base_dir = Path(base_directory)
    task_idx_to_path = {}
    for buffer_path in glob.glob(str(base_dir / 'macaw_buffer*')):
        pattern = re.compile('macaw_buffer_task_(\d+).npy')
        match = pattern.search(buffer_path)
        task_idx = int(match.group(1))
        task_idx_to_path[task_idx] = buffer_path

    for task_idx in train_task_idxs:
        dataset_path = task_idx_to_path[task_idx]
        data = np.load(dataset_path, allow_pickle=True).item()
        algo.replay_buffer.task_buffers[task_idx].reinitialize_from_dict(
            data,
            start_idx=start_idx,
            end_idx=end_idx,
        )
        algo.enc_replay_buffer.task_buffers[task_idx].reinitialize_from_dict(
            data,
            start_idx=start_idx,
            end_idx=end_idx,
        )


def load_buffer_onto_algo(
        algo: MetaRLAlgorithm,
        pretrain_buffer_path: str,
        start_idx=0,
        end_idx=None,
        start_idx_enc=0,
        end_idx_enc=None,
):
    data = load_local_or_remote_file(
        pretrain_buffer_path,
        file_type='joblib',
    )
    saved_replay_buffer = data['replay_buffer']
    saved_enc_replay_buffer = data['enc_replay_buffer']
    if algo.use_meta_learning_buffer:
        for k in saved_replay_buffer.task_buffers:
            if k not in saved_replay_buffer.task_buffers:
                print("No saved buffer for task {}. Skipping.".format(k))
                continue
            new_buffer = algo.meta_replay_buffer.create_buffer()
            new_buffer.copy_data(
                saved_replay_buffer.task_buffers[k],
                start_idx=start_idx,
                end_idx=end_idx,
            )
            algo.meta_replay_buffer.append_buffer(new_buffer)
    else:
        rl_replay_buffer = algo.replay_buffer
        encoder_replay_buffer = algo.enc_replay_buffer
        if algo.encoder_buffer_matches_rl_buffer:
            saved_enc_replay_buffer = saved_replay_buffer
        for k in rl_replay_buffer.task_buffers:
            if k not in saved_replay_buffer.task_buffers:
                print("No saved buffer for task {}. Skipping.".format(k))
                continue
            rl_replay_buffer.task_buffers[k].copy_data(
                saved_replay_buffer.task_buffers[k],
                start_idx=start_idx,
                end_idx=end_idx,
            )
        for k in encoder_replay_buffer.task_buffers:
            if k not in saved_enc_replay_buffer.task_buffers:
                print("No saved buffer for task {}. Skipping.".format(k))
                continue
            encoder_replay_buffer.task_buffers[k].copy_data(
                saved_enc_replay_buffer.task_buffers[k],
                start_idx=start_idx_enc,
                end_idx=end_idx_enc,
            )


class EvalPearl(object):
    def __init__(
            self,
            algorithm: MetaRLAlgorithm,
            train_task_indices: List[int],
            test_task_indices: List[int],
    ):
        self.algorithm = algorithm
        self.train_task_indices = train_task_indices
        self.test_task_indices = test_task_indices

    def __call__(self):
        results = OrderedDict()
        for name, indices in [
            ('train_tasks', self.train_task_indices),
            ('test_tasks', self.test_task_indices),
        ]:
            final_returns, online_returns = self.algorithm._do_eval(indices, -1)
            results['eval/adaptation/{}/final_returns Mean'.format(name)] = np.mean(final_returns)
            results['eval/adaptation/{}/all_returns Mean'.format(name)] = np.mean(online_returns)

            paths = []
        for idx in self.train_task_indices:
            paths += self._get_init_from_buffer_path(idx)
        results['eval/init_from_buffer/train_tasks/all_returns Mean'] = np.mean(
            eval_util.get_average_returns(paths)
        )
        return results

    def _get_init_from_buffer_path(self, idx):
        if self.algorithm.use_meta_learning_buffer:
            init_context = self.algorithm.meta_replay_buffer._sample_contexts(
                [idx],
                self.algorithm.embedding_batch_size
            )
        else:
            init_context = self.algorithm.enc_replay_buffer.sample_context(
                idx,
                self.algorithm.embedding_batch_size
            )
        init_context = ptu.from_numpy(init_context)
        p, _ = self.algorithm.sampler.obtain_samples(
            deterministic=self.algorithm.eval_deterministic,
            max_samples=self.algorithm.max_path_length,
            accum_context=False,
            max_trajs=1,
            resample_latent_period=0,
            update_posterior_period=0,
            initial_context=init_context,
            task_idx=idx,
        )
        return p
