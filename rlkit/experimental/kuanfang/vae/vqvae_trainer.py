from collections import OrderedDict
from os import path as osp
import numpy as np
import torch
from rlkit.core.loss import LossFunction
from torch import optim
from torchvision.utils import save_image
from rlkit.data_management.images import normalize_image
from rlkit.core import logger
from rlkit.torch import pytorch_util as ptu
import collections
import time


class VqVaeTrainer(LossFunction):

    def __init__(
            self,
            model,
            batch_size=128,
            log_interval=0,
            lr=1e-3,
            normalize=False,
            mse_weight=0.1,
            is_auto_encoder=False,
            background_subtract=False,
            linearity_weight=0.0,
            distance_weight=0.0,
            loss_weights=None,
            train_data_workers=2,
            priority_function_kwargs=None,
            weight_decay=0,
            num_epochs=None,
            tf_logger=None,
    ):
        self.log_interval = log_interval
        self.batch_size = batch_size

        self.num_epochs = num_epochs
        self.imsize = model.imsize
        model.to(ptu.device)

        self.model = model
        self.representation_size = model.representation_size
        self.input_channels = model.input_channels
        self.imlength = model.imlength

        self.lr = lr
        params = list(self.model.parameters())
        self.optimizer = optim.Adam(params,
                                    lr=self.lr,
                                    weight_decay=weight_decay)

        self.train_data_workers = train_data_workers

        if priority_function_kwargs is None:
            self.priority_function_kwargs = dict()
        else:
            self.priority_function_kwargs = priority_function_kwargs

        self.normalize = normalize
        self.background_subtract = background_subtract
        if self.normalize or self.background_subtract:
            self.train_data_mean = np.mean(self.train_dataset, axis=0)
            self.train_data_mean = normalize_image(
                np.uint8(self.train_data_mean)
            )

        self.linearity_weight = linearity_weight
        self.distance_weight = distance_weight
        self.loss_weights = loss_weights

        # stateful tracking variables, reset every epoch
        self.eval_statistics = collections.defaultdict(list)
        self.eval_data = collections.defaultdict(list)
        self.num_batches = 0

        self.tf_logger = tf_logger

    @property
    def log_dir(self):
        return logger.get_snapshot_dir()

    def end_epoch(self, epoch):
        self.eval_statistics = collections.defaultdict(list)
        self.test_last_batch = None

    def get_diagnostics(self):
        stats = OrderedDict()
        for k in sorted(self.eval_statistics.keys()):
            stats[k] = np.mean(self.eval_statistics[k])
        return stats

    def train_epoch(self, epoch, dataloader, batches=100):
        start_time = time.time()
        for b in range(batches):
            batch = next(iter(dataloader))
            self.train_batch(epoch, batch)
            # self.train_batch(epoch, dataset.random_batch(self.batch_size))
        self.eval_statistics['train/epoch_duration'].append(
            time.time() - start_time)

    def test_epoch(self, epoch, dataloader, batches=10):
        start_time = time.time()
        for b in range(batches):
            batch = next(iter(dataloader))
            self.test_batch(epoch, batch)
            # self.test_batch(epoch, dataset.random_batch(self.batch_size))
        self.eval_statistics['test/epoch_duration'].append(
            time.time() - start_time)

    def train_batch(self, epoch, batch):
        self.num_batches += 1
        # self.model.train()
        self.optimizer.zero_grad()

        loss = self.compute_loss(batch, epoch, 'train')
        loss.backward()
        self.optimizer.step()
        # self.scheduler.step()

    def test_batch(
            self,
            epoch,
            batch,
    ):
        self.model.eval()
        loss = self.compute_loss(batch, epoch, 'test')  # NOQA

    def compute_loss(self, batch, epoch, prefix):
        loss, extra = self.model.compute_loss(batch)

        self.eval_statistics['epoch'] = epoch
        self.eval_statistics['num_batches'] = self.num_batches
        self.eval_statistics['%s/%s' % (prefix, 'loss')].append(loss.item())

        for key in ['loss_vq', 'loss_recon', 'perplexity']:
            value = extra[key]
            self.eval_statistics['%s/%s' % (prefix, key)].append(
                value.item())

        # for key, value in extra.items():
        #     self.eval_statistics['%s/%s' % (prefix, key)].append(
        #         value.item())

        batch_recon = extra['recon']
        for i in range(batch.shape[-1]):
            self.tf_logger.log_histogram(
                '%s/x/dim_%d' % (prefix, i),
                batch[..., i].detach(),
                epoch)
            self.tf_logger.log_histogram(
                '%s/x_recon/dim_%d' % (prefix, i),
                batch_recon[..., i].detach(),
                epoch)

        self.tf_logger.log_images(
            '%s_image_input' % (prefix),
            batch[:4].detach() + 0.5,
            epoch)

        self.tf_logger.log_images(
            '%s_image_recon' % (prefix),
            batch_recon[:4].detach() + 0.5,
            epoch)

        return loss

        # def encode_dataset(self, dataset):
        #     encoding_list = []
        #     save_dir = osp.join(self.log_dir, 'dataset_latents.npy')
        #     for i in range(len(dataset)):
        #         batch = dataset.random_batch(self.batch_size)
        #         obs, cond = batch['x_t'], batch['env']
        #         z_delta = self.model.encode(obs, cont=False)
        #         z_cond = self.model.encode(cond, cont=False)
        #         encodings = torch.cat([z_delta, z_cond], dim=1)
        #         encoding_list.append(encodings)
        #     encodings = ptu.get_numpy(torch.cat(encoding_list))
        #     np.save(save_dir, encodings)

        # def get_dataset_stats(self, data):
        #     torch_input = ptu.from_numpy(normalize_image(data))
        #     mus, log_vars = self.model.encode(torch_input)
        #     mus = ptu.get_numpy(mus)
        #     mean = np.mean(mus, axis=0)
        #     std = np.std(mus, axis=0)
        #     return mus, mean, std

        def dump_reconstructions(self, epoch):
            obs, reconstructions = self.eval_data['test/last_batch']
            n = min(obs.size(0), 8)
            comparison = torch.cat([
                obs[:n].narrow(
                    start=0, length=self.imlength, dim=1)
                .contiguous().view(
                    -1, self.input_channels, self.imsize, self.imsize
                ).transpose(2, 3),
                reconstructions.view(
                    self.batch_size,
                    self.input_channels,
                    self.imsize,
                    self.imsize,
                )[:n].transpose(2, 3)
            ])
            save_dir = osp.join(
                self.log_dir, 'test_recon_%d.png' % epoch)
            save_image(comparison.data.cpu(), save_dir, nrow=n)

            obs, reconstructions = self.eval_data['train/last_batch']
            n = min(obs.size(0), 8)
            comparison = torch.cat([
                obs[:n].narrow(
                    start=0, length=self.imlength, dim=1)
                .contiguous().view(
                    -1, self.input_channels, self.imsize, self.imsize
                ).transpose(2, 3),
                reconstructions.view(
                    self.batch_size,
                    self.input_channels,
                    self.imsize,
                    self.imsize,
                )[:n].transpose(2, 3)
            ])
            save_dir = osp.join(
                self.log_dir, 'train_recon_%d.png' % epoch)
            save_image(comparison.data.cpu(), save_dir, nrow=n)

        def dump_samples(self, epoch): return
